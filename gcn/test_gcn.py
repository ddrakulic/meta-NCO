import argparse
import warnings
import os
import torch

from plots.make_plots import generate_plots
import numpy as np
from supervisors.handlers.gcn_tsp.task import TaskHandler

# Remove warning
from supervisors.test_supervisor import run_test
from utils import load_config

warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='meta_gcn_tsp')
    parser.add_argument('--train_type', type=str, default="reptile")
    parser.add_argument('--model', type=str, default="gcn_tsp")
    parser.add_argument('--task_type', type=str, default="size")
    parser.add_argument('--output_dir', type=str, default="./outputs")
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--job_id', type=str, default=1)
    parser.add_argument("--train_task_list", type=int, nargs='+', default=[], help="Training tasks")
    parser.add_argument("--test_task_list", type=int, nargs='+', default=[20], help="Testing tasks")
    parser.add_argument('--trained_model', type=str,
                        default="./outputs/meta_nco/gcn_tsp/reptile/size/359953/best_val_checkpoint.tar")
    parser.add_argument('--seed', type=int, default=0)
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    assert (args.task_type == "mix" or args.task_type == "mode" or args.task_type == "scale" or args.task_type == "size")

    working_dir = os.path.dirname(os.path.abspath(__file__))
    config_net = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "gcn_tsp",
                                          "config_net.json"))

    handler = TaskHandler(args, config_net)

    print("Agrs:", args)
    log_dir = f"{args.output_dir}/meta_nco/gcn_tsp/{args.train_type}/{args.task_type}/{args.job_id}/"
    os.makedirs(log_dir, exist_ok=True)
    run_test(handler, args.test_dataset_size, log_dir, args.trained_model)

    generate_plots(log_dir, args.job_id, args.test_task_list, train_logs=False)

