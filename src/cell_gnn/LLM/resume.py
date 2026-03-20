import os
import re


def detect_last_iteration(analysis_path, config_save_dir, n_parallel):
    """Detect the last fully completed batch from saved artifacts."""
    found_iters = set()

    if os.path.exists(analysis_path):
        with open(analysis_path, 'r') as f:
            for line in f:
                match = re.match(r'^##+ Iter (\d+):', line)
                if match:
                    found_iters.add(int(match.group(1)))

    if os.path.isdir(config_save_dir):
        for fname in os.listdir(config_save_dir):
            match = re.match(r'iter_(\d+)_slot_\d+\.yaml', fname)
            if match:
                found_iters.add(int(match.group(1)))

    if not found_iters:
        return 1

    last_iter = max(found_iters)
    batch_start = ((last_iter - 1) // n_parallel) * n_parallel + 1
    batch_iters = set(range(batch_start, batch_start + n_parallel))

    if batch_iters.issubset(found_iters):
        resume_at = batch_start + n_parallel
    else:
        resume_at = batch_start

    return resume_at
