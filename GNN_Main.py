import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")

from particle_gnn.config import ParticleGNNConfig
from particle_gnn.generators.graph_data_generator import data_generate
from particle_gnn.models.graph_trainer import data_train, data_test
from particle_gnn.utils import set_device, add_pre_folder

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    parser = argparse.ArgumentParser(description="ParticleGNN")
    parser.add_argument("-o", "--option", nargs="+", help="Option that takes multiple values")

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
        config = ParticleGNNConfig.from_yaml(f"{config_root}/{config_file}.yaml")
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_file_
        device = set_device(config.training.device)

        print(f"config_file  {config.config_file}")
        print(f"\033[92mdevice  {device}\033[0m")

        if "generate" in task:
            data_generate(
                config,
                device=device,
                visualize=True,
                run_vizualized=0,
                style="color",
                alpha=1,
                erase=True,
                bSave=True,
                step=100,
                timer=False
            )

        if "train" in task:
            data_train(config=config, erase=False, best_model=best_model, device=device)

        if "test" in task:
            data_test(
                config=config,
                visualize=True,
                style="black color name",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="",
                sample_embedding=False,
                step=5,
                device=device,
                particle_of_interest=0,
            )
