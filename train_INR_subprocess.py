"""Standalone entry point for training an INR on a cell-gnn field.

Usage:
    python train_INR_subprocess.py <config_name> [--field <field_name>] [--run <run>] [--erase]

Examples:
    python train_INR_subprocess.py misc/dicty
    python train_INR_subprocess.py misc/dicty --field residual --erase
    python train_INR_subprocess.py misc/dicty --field velocity --run 1
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")

from cell_gnn.config import CellGNNConfig
from cell_gnn.utils import set_device, add_pre_folder
from cell_gnn.models.inr_trainer import data_train_INR

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    parser = argparse.ArgumentParser(description="Train INR on a cell-gnn field")
    parser.add_argument("config", type=str, help="Config name (e.g. misc/dicty)")
    parser.add_argument("--field", type=str, default=None,
                        help="Field name to fit (default: from config or 'residual')")
    parser.add_argument("--run", type=int, default=0, help="Dataset run index")
    parser.add_argument("--erase", action="store_true", help="Erase previous INR outputs")
    args = parser.parse_args()

    config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"
    config_file, pre_folder = add_pre_folder(args.config)
    config = CellGNNConfig.from_yaml(f"{config_root}/{config_file}.yaml")
    config.dataset = pre_folder + config.dataset
    config.config_file = pre_folder + args.config
    device = set_device(config.training.device)

    print(f"config_file  {config.config_file}")
    print(f"\033[92mdevice  {device}\033[0m")

    field_name = args.field
    if field_name is None:
        field_name = config.inr.inr_field_name if config.inr else 'residual'

    data_train_INR(
        config=config,
        device=device,
        field_name=field_name,
        run=args.run,
        erase=args.erase,
    )
