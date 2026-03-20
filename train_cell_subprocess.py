#!/usr/bin/env python3
"""
Standalone cell-gnn training+test script for subprocess execution.

This script is called by GNN_LLM_parallel.py as a subprocess to ensure that any code
modifications to graph_trainer.py are reloaded for each iteration.

Usage:
    python train_cell_subprocess.py --config CONFIG_PATH --device DEVICE [--erase] [--log_file LOG_PATH]
"""

import matplotlib
matplotlib.use('Agg')  # set non-interactive backend before other imports

import argparse
import sys
import os
import traceback

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from cell_gnn.config import CellGNNConfig
from cell_gnn.models.graph_trainer import data_train
from cell_gnn.utils import set_device


def main():
    parser = argparse.ArgumentParser(description='Train+test cell-gnn')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--erase', action='store_true', help='Erase existing log files')
    parser.add_argument('--log_file', type=str, default=None, help='Path to analysis log file')
    parser.add_argument('--config_file', type=str, default=None, help='Config file name for log directory')
    parser.add_argument('--error_log', type=str, default=None, help='Path to error log file')
    parser.add_argument('--generate', action='store_true', help='Regenerate data before training')

    args = parser.parse_args()

    # Open error log file if specified
    error_log = None
    if args.error_log:
        try:
            error_log = open(args.error_log, 'w')
        except Exception as e:
            print(f"Warning: Could not open error log file: {e}", file=sys.stderr)

    try:
        # Load config
        config = CellGNNConfig.from_yaml(args.config)

        # Set config_file if provided (needed for proper log directory path)
        if args.config_file:
            config.config_file = args.config_file
            pre_folder = os.path.dirname(args.config_file)
            if pre_folder:
                pre_folder += '/'
            if not config.dataset.startswith(pre_folder):
                config.dataset = pre_folder + config.dataset

        # Set device
        device = set_device(args.device)

        # Phase 0: Generate data (if requested)
        if args.generate:
            from cell_gnn.generators.graph_data_generator import data_generate
            print("Generating data ...")
            data_generate(
                config=config,
                device=device,
                visualize=False,
                run_vizualized=0,
                style="color",
                alpha=1,
                erase=True,
                save=True,
                step=50,
            )

        # Open log file if specified
        log_file = None
        if args.log_file:
            log_file = open(args.log_file, 'w')

        try:
            # Train only — test runs locally after cluster job completes
            data_train(
                config=config,
                erase=args.erase,
                best_model=None,
                device=device,
                log_file=log_file,
            )

        finally:
            if log_file:
                log_file.close()

    except Exception as e:
        # Capture full traceback for debugging
        error_msg = f"\n{'='*80}\n"
        error_msg += "TRAINING SUBPROCESS ERROR\n"
        error_msg += f"{'='*80}\n\n"
        error_msg += f"Error Type: {type(e).__name__}\n"
        error_msg += f"Error Message: {str(e)}\n\n"
        error_msg += "Full Traceback:\n"
        error_msg += traceback.format_exc()
        error_msg += f"\n{'='*80}\n"

        # Print to stderr
        print(error_msg, file=sys.stderr, flush=True)

        # Write to error log if available
        if error_log:
            error_log.write(error_msg)
            error_log.flush()

        # Exit with non-zero code
        sys.exit(1)

    finally:
        if error_log:
            error_log.close()


if __name__ == '__main__':
    main()
