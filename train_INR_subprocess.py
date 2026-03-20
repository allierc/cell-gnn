#!/usr/bin/env python3
"""Standalone entry point for training an INR on a cell-gnn field.

Supports two modes:
  1. Direct: python train_INR_subprocess.py misc/dicty [--field velocity] [--erase]
  2. Parallel (used by GNN_LLM_INR_parallel.py):
     python train_INR_subprocess.py --config CONFIG_PATH --device cuda
         --log_file LOG_PATH --config_file CONFIG_FILE [--erase]
"""

import argparse
import os
import sys
import traceback
import warnings

import matplotlib
matplotlib.use("Agg")

from cell_gnn.config import CellGNNConfig
from cell_gnn.utils import set_device, add_pre_folder, config_path, log_path
from cell_gnn.models.inr_trainer import data_train_INR

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    parser = argparse.ArgumentParser(description="Train INR on a cell-gnn field")
    # Direct mode (positional)
    parser.add_argument("config_name", type=str, nargs='?', default=None,
                        help="Config name for direct mode (e.g. misc/dicty)")
    # Parallel mode (keyword)
    parser.add_argument("--config", type=str, default=None,
                        help="Full path to config YAML (parallel mode)")
    parser.add_argument("--config_file", type=str, default=None,
                        help="Config file name for log directory (parallel mode)")
    parser.add_argument("--device", type=str, default='auto', help="Device to use")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to analysis log file (parallel mode)")
    parser.add_argument("--error_log", type=str, default=None,
                        help="Path to error log file")
    parser.add_argument("--field", type=str, default=None,
                        help="Field name to fit (default: from config)")
    parser.add_argument("--run", type=int, default=0, help="Dataset run index")
    parser.add_argument("--erase", action="store_true", help="Erase previous INR outputs")
    args = parser.parse_args()

    error_log = None
    if args.error_log:
        try:
            error_log = open(args.error_log, 'w')
        except Exception as e:
            print(f"Warning: Could not open error log file: {e}", file=sys.stderr)

    try:
        # Determine mode
        if args.config is not None:
            # Parallel mode: full path to config
            config = CellGNNConfig.from_yaml(args.config)
            if args.config_file:
                config.config_file = args.config_file
                pre_folder = os.path.dirname(args.config_file)
                if pre_folder:
                    pre_folder += '/'
                config.dataset = pre_folder + config.dataset
        elif args.config_name is not None:
            # Direct mode: config name
            config_file, pre_folder = add_pre_folder(args.config_name)
            config = CellGNNConfig.from_yaml(config_path(f"{config_file}.yaml"))
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + args.config_name
        else:
            print("Error: provide either a config name or --config path", file=sys.stderr)
            sys.exit(1)

        device = set_device(args.device)
        print(f"config_file  {config.config_file}")
        print(f"\033[92mdevice  {device}\033[0m")

        field_name = args.field
        if field_name is None:
            field_name = config.inr.inr_field_name if config.inr else 'velocity'

        model, loss_list = data_train_INR(
            config=config,
            device=device,
            field_name=field_name,
            run=args.run,
            erase=args.erase,
        )

        # If log_file specified, copy results.log content there for the parallel harness
        if args.log_file:
            log_dir = log_path(config.config_file)
            results_path = f'./{log_dir}/tmp_training/inr/results.log'
            if os.path.exists(results_path):
                with open(results_path, 'r') as rf:
                    results_content = rf.read()
                with open(args.log_file, 'w') as lf:
                    lf.write(results_content)
                print(f"  results copied to {args.log_file}")

    except Exception as e:
        error_msg = f"\n{'='*80}\n"
        error_msg += "INR TRAINING SUBPROCESS ERROR\n"
        error_msg += f"{'='*80}\n\n"
        error_msg += f"Error Type: {type(e).__name__}\n"
        error_msg += f"Error Message: {str(e)}\n\n"
        error_msg += "Full Traceback:\n"
        error_msg += traceback.format_exc()
        error_msg += f"\n{'='*80}\n"

        print(error_msg, file=sys.stderr, flush=True)
        if error_log:
            error_log.write(error_msg)
            error_log.flush()
        sys.exit(1)

    finally:
        if error_log:
            error_log.close()
