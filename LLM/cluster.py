import os
import re
import subprocess
import time

# ---------------------------------------------------------------------------
# Cluster constants
# ---------------------------------------------------------------------------

CLUSTER_USER = "allierc"
CLUSTER_LOGIN = "login1"
CLUSTER_HOME = "/groups/saalfeld/home/allierc"
CLUSTER_ROOT_DIR = f"{CLUSTER_HOME}/GraphCluster/cell-gnn"
CLUSTER_DATA_DIR = f"{CLUSTER_HOME}/GraphData"
CLUSTER_SSH = f"{CLUSTER_USER}@{CLUSTER_LOGIN}"


# ---------------------------------------------------------------------------
# Cluster helpers
# ---------------------------------------------------------------------------

def local_to_cluster(path: str, root_dir: str) -> str:
    """Map a local workspace path to the correct cluster path.

    Mounted directories (config, log, graphs_data) live under GraphData on the
    cluster, not under GraphCluster/cell-gnn. Everything else maps to CLUSTER_ROOT_DIR.
    """
    for sub in ('config', 'log', 'graphs_data'):
        local_sub = os.path.join(root_dir, sub)
        if path.startswith(local_sub):
            return os.path.join(CLUSTER_DATA_DIR, sub) + path[len(local_sub):]
    return path.replace(root_dir, CLUSTER_ROOT_DIR)


def check_cluster_repo():
    """Check that GraphCluster/cell-gnn has no uncommitted source changes.

    Runs `git diff HEAD` on the cluster via SSH, excluding config/ (which is
    expected to be modified by the LLM).  Returns True if clean, False if dirty.
    """
    ssh_cmd = (
        f"ssh {CLUSTER_SSH} "
        f"\"cd {CLUSTER_ROOT_DIR} && git diff HEAD --stat -- . ':!config/'\""
    )
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)
    diff_output = result.stdout.strip()
    if diff_output:
        print("\033[91mERROR: GraphCluster repo has uncommitted changes — commit and push before running\033[0m")
        for line in diff_output.splitlines():
            print(f"  \033[91m{line}\033[0m")
        return False
    print("\033[92mCluster repo: git diff clean (no uncommitted source changes)\033[0m")
    return True


def submit_cluster_job(slot, config_path, analysis_log_path, config_file_field,
                       log_dir, root_dir, erase=True, node_name='a100',
                       exploration_dir=None, iteration=None):
    """Submit a single cell-gnn training job to the cluster WITHOUT -K (non-blocking).

    Data generation and test/plot are handled locally.
    The cluster job runs training only.
    """
    cluster_script_path = f"{log_dir}/cluster_train_{slot:02d}.sh"
    error_details_path = f"{log_dir}/training_error_{slot:02d}.log"

    cluster_config_path = local_to_cluster(config_path, root_dir)
    cluster_analysis_log = local_to_cluster(analysis_log_path, root_dir)
    cluster_error_log = local_to_cluster(error_details_path, root_dir)

    cluster_train_cmd = f"python train_cell_subprocess.py --config '{cluster_config_path}' --device cuda"
    cluster_train_cmd += f" --log_file '{cluster_analysis_log}'"
    cluster_train_cmd += f" --config_file '{config_file_field}'"
    cluster_train_cmd += f" --error_log '{cluster_error_log}'"
    if erase:
        cluster_train_cmd += " --erase"
    if exploration_dir is not None and iteration is not None:
        cluster_exploration_dir = local_to_cluster(exploration_dir, root_dir)
        cluster_train_cmd += f" --exploration_dir '{cluster_exploration_dir}'"
        cluster_train_cmd += f" --iteration {iteration}"
        cluster_train_cmd += f" --slot {slot}"

    with open(cluster_script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"cd {CLUSTER_ROOT_DIR}\n")
        f.write(f"conda run -n cell-gnn {cluster_train_cmd}\n")
    os.chmod(cluster_script_path, 0o755)

    cluster_script = local_to_cluster(cluster_script_path, root_dir)
    cluster_log_dir = local_to_cluster(log_dir, root_dir)
    cluster_stdout = f"{cluster_log_dir}/cluster_train_{slot:02d}.out"
    cluster_stderr = f"{cluster_log_dir}/cluster_train_{slot:02d}.err"

    ssh_cmd = (
        f"ssh {CLUSTER_SSH} \"cd {CLUSTER_ROOT_DIR} && "
        f"bsub -n 8 -gpu 'num=1' -q gpu_{node_name} -W 6000 "
        f"-o '{cluster_stdout}' -e '{cluster_stderr}' "
        f"'bash {cluster_script}'\""
    )
    print(f"\033[96m  slot {slot}: submitting via SSH\033[0m", flush=True)
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)

    match = re.search(r'Job <(\d+)>', result.stdout)
    if match:
        job_id = match.group(1)
        print(f"\033[92m  slot {slot}: job {job_id} submitted\033[0m")
        return job_id
    else:
        print(f"\033[91m  slot {slot}: submission FAILED\033[0m")
        print(f"    stdout: {result.stdout.strip()}")
        print(f"    stderr: {result.stderr.strip()}")
        return None


def wait_for_cluster_jobs(job_ids, log_dir=None, poll_interval=60):
    """Poll bjobs via SSH until all jobs finish."""
    pending = dict(job_ids)
    results = {}

    while pending:
        ids_str = ' '.join(pending.values())
        ssh_cmd = f'ssh {CLUSTER_SSH} "bjobs {ids_str} 2>/dev/null"'
        out = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)

        for slot, jid in list(pending.items()):
            for line in out.stdout.splitlines():
                if jid in line:
                    if 'DONE' in line:
                        results[slot] = True
                        del pending[slot]
                        print(f"\033[92m  slot {slot} (job {jid}): DONE\033[0m")
                    elif 'EXIT' in line:
                        results[slot] = False
                        del pending[slot]
                        print(f"\033[91m  slot {slot} (job {jid}): FAILED (EXIT)\033[0m")
                        if log_dir:
                            err_file = f"{log_dir}/cluster_train_{slot:02d}.err"
                            if os.path.exists(err_file):
                                try:
                                    with open(err_file, 'r') as ef:
                                        err_content = ef.read().strip()
                                    if err_content:
                                        print(f"\033[91m  --- slot {slot} error log ---\033[0m")
                                        for eline in err_content.splitlines()[-30:]:
                                            print(f"\033[91m    {eline}\033[0m")
                                        print("\033[91m  --- end error log ---\033[0m")
                                except Exception:
                                    pass

            if slot in pending and jid not in out.stdout:
                results[slot] = True
                del pending[slot]
                print(f"\033[93m  slot {slot} (job {jid}): no longer in queue (assuming DONE)\033[0m")

        if pending:
            statuses = [f"slot {s}" for s in pending]
            print(f"\033[90m  ... waiting for {', '.join(statuses)} ({poll_interval}s)\033[0m")
            time.sleep(poll_interval)

    return results
