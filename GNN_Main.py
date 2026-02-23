import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")

from cell_gnn.config import CellGNNConfig
from cell_gnn.generators.graph_data_generator import data_generate
from cell_gnn.models.graph_trainer import data_train, data_test
from cell_gnn.utils import set_device, add_pre_folder

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    parser = argparse.ArgumentParser(description="CellGNN")
    parser.add_argument("-o", "--option", nargs="+", help="Option that takes multiple values")
    parser.add_argument("--n_epochs", type=int, default=None, help="Override n_epochs from config")
    parser.add_argument("--erase", action="store_true", help="Erase previous training results")

    args = parser.parse_args()

    if args.option:
        print(f"Options: {args.option}")
    if args.option is not None:
        task = args.option[0]
        config_list = [args.option[1]]
        if len(args.option) > 2:
            best_model = args.option[2]
        else:
            best_model = None
    else:
        best_model = None
        task = 'generate'
        config_list = ['arbitrary_3']

    for config_file_ in config_list:
        print(" ")
        config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"
        config_file, pre_folder = add_pre_folder(config_file_)
        config = CellGNNConfig.from_yaml(f"{config_root}/{config_file}.yaml")
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_file_
        if args.n_epochs is not None:
            config.training.n_epochs = args.n_epochs
            config.training.small_init_batch_size = False
        device = set_device(config.training.device)

        print(f"config_file  {config.config_file}")
        print(f"\033[92mdevice  {device}\033[0m")

        if "generate" in task:
            data_generate(
                config,
                device=device,
                visualize=True,
                run_vizualized=0,
                style="color edges",
                alpha=1,
                erase=True,
                save=True,
                step=50,
                timer=False
            )

        if task == "train":
            data_train(config=config, erase=args.erase, best_model=best_model, device=device)

        if "train_inr" in task:
            from cell_gnn.models.inr_trainer import data_train_INR
            data_train_INR(
                config=config,
                device=device,
                field_name=config.inr.inr_field_name if config.inr else 'residual',
                run=0,
                erase=args.erase,
            )

        if "test" in task:
            data_test(
                config=config,
                visualize=True,
                style="color residual true",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="",
                sample_embedding=False,
                step=250,
                device=device,
                cell_of_interest=0,
            )
