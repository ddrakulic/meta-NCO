import argparse
import os
import random
import warnings
import numpy as np
import torch
from utils import load_config
from supervisors import reptile_supervisor
from plots.make_plots import generate_plots
from supervisors.handlers.gcn_tsp.task import TaskHandler

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='meta_gcn_tsp')
    parser.add_argument("--task_type", type=str, default="size", help="mode/size/scale/mix")
    parser.add_argument("--train_task_list", type=int, nargs='+', default=[10], help="Training tasks")
    parser.add_argument("--test_task_list", type=int, nargs='+', default=[20], help="Testing tasks")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--job_id", type=str, default=1)
    # training params
    parser.add_argument("--alpha", type=float, default=0.999)
    parser.add_argument("--decay_alpha", type=float, default=0.995)
    parser.add_argument("--time_limit", type=float, default=24, help="training time limit, in hours")
    parser.add_argument("--val_every", type=int, default=5, help="Validate after every n epoch")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument("--task_learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--decay_learning_rate", type=float, default=0.99, help="Decay rate for learning rate")
    parser.add_argument("--train_steps_per_epoch", type=int, default=1, help="Training steps per epoch")
    parser.add_argument("--train_finetuning_steps", type=int, default=50, help="Training finetuning steps")
    parser.add_argument("--test_dataset_size", type=float, default=0.5, help="Test dataset percent")
    parser.add_argument("--test_finetuning_steps", type=int, default=52, help="Test finetuning steps")
    parser.add_argument("--test_every_n_finetuning_steps", type=int, default=2, help="Test every n finetuning steps")
    parser.add_argument("--test_finetuning_learning_rate", type=float, default=0.0001,
                        help="Test finetuning learning rate")

    args = parser.parse_args()

    assert args.task_type == "mode" or args.task_type == "scale" or args.task_type == "size" or args.task_type == "mix"

    print("Task:", args.task_type)
    print("SEED:", args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_net = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "configs", "gcn_tsp", "config_net.json"))

    print("Program arguments", args)
    print("Network parameters", config_net)

    # log directories
    log_dir = f"{args.output_dir}/meta_nco/gcn_tsp/reptile/{args.task_type}/{args.job_id}/"

    handler = TaskHandler(args, config_net)
    supervisor = reptile_supervisor.Supervisor(args, handler, log_dir)

    supervisor.train()
    generate_plots(supervisor.log_dir, args.job_id, args.test_task_list)
