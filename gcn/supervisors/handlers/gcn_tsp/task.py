import copy
import torch
from torch import nn
import os
from base_models.gcn_tsp.models.gcn_model import ResidualGatedGCNModel
from base_models.gcn_tsp.utils.google_tsp_reader import GoogleTSPReader
from base_models.gcn_tsp.utils.graph_utils import mean_tour_len_edges, mean_tour_len_nodes
from base_models.gcn_tsp.utils.model_utils import beamsearch_tour_nodes, beamsearch_tour_nodes_shortest
from utils import DotDict
from supervisors.handlers.base import TaskHandlerBase

MODEL = "gcn_tsp"


class TaskHandler(TaskHandlerBase):
    def __init__(self, args, config_net):
        # make the network
        self.net = nn.DataParallel(ResidualGatedGCNModel(config_net))
        if torch.cuda.is_available():
            self.net = self.net.cuda()

        self.args = args

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.task_learning_rate)

        # load datasets
        self.support_dataset, self.query_dataset = dict(), dict()

        # load task_configs
        self.config_tasks, self.running_data, self.datasets = dict(), dict(), dict()

        for task in set(args.train_task_list).union(args.test_task_list):
            if args.task_type == "size":
                self.config_tasks[task] = self.make_config(args.data_dir, size=task, task_type=args.task_type)
            elif args.task_type == "mode":
                self.config_tasks[task] = self.make_config(args.data_dir, size=50, modes=task, task_type=args.task_type)
            elif args.task_type == "scale":
                self.config_tasks[task] = self.make_config(args.data_dir, size=50, scale=task, task_type=args.task_type)
            else:
                # it is complicated with mix, task 204 = size 20, modes 4
                size = int(str(task)[:-1])
                modes = int(str(task)[-1])
                self.config_tasks[task] = self.make_config(args.data_dir, size=size, modes=modes, task_type=args.task_type)

    def forward(self, task, action="train"):
        # forward pass
        self.net.train()
        inputs = next(self.support_dataset[task])

        outputs, loss = self.net(inputs)
        loss = loss.mean()
        if action == "train":
            # update running data
            pred_tour_len = mean_tour_len_edges(inputs["edges_values"], outputs)
            gt_tour_len = torch.mean(inputs["tour_len"])
            self.running_data[task]["loss"] += loss.data.item()
            self.running_data[task]["pred_tour_len"] += pred_tour_len
            self.running_data[task]["gt_tour_len"] += gt_tour_len.item()
        return loss

    def evaluate(self, task, action, step=0):
        # evaluation
        if action == "validation":
            query_dataset = iter(GoogleTSPReader(self.config_tasks[task].num_nodes,
                                                 self.config_tasks[task].num_neighbors,
                                                 self.args.batch_size,
                                                 self.config_tasks[task].val_filepath,
                                                 self.config_tasks[task].val_size,
                                                 shuffle=False))
        else:
            query_dataset = iter(GoogleTSPReader(self.config_tasks[task].num_nodes,
                                                 self.config_tasks[task].num_neighbors,
                                                 self.args.batch_size,
                                                 self.config_tasks[task].test_filepath,
                                                 self.config_tasks[task].test_size,
                                                 # we are testing on 5K instances = half of test dataset
                                                 split_details={"percentage": 0.5, "part": 0},
                                                 shuffle=False))
        total_loss, total_opt_gap = 0., 0.
        tour_lens = dict()
        tour_lens["pred_tour_len"] = 0.
        tour_lens["gt_tour_len"] = 0.

        self.net.eval()
        with torch.no_grad():
            num_instances = 0
            for inputs in query_dataset:
                # Forward pass
                outputs, loss = self.net(inputs)
                loss = loss.mean()  # Take mean of loss across multiple GPUs
                # final_evaluation = updates current running data and returns gap
                opt_gap, pred_tour_len, gt_tour_len, opt_gaps, bs_tours = self._final_evaluation(task, inputs, outputs,
                                                                                                 action)
                total_opt_gap += opt_gap
                total_loss += loss.item()
                tour_lens["pred_tour_len"] += pred_tour_len
                tour_lens["gt_tour_len"] += gt_tour_len
                num_instances += 1

        total_opt_gap /= num_instances
        total_loss /= num_instances
        tour_lens["pred_tour_len"] /= num_instances
        tour_lens["gt_tour_len"] /= num_instances

        return 100 * total_opt_gap, total_loss, tour_lens

    def _final_evaluation(self, task, inputs, outputs, mode):
        if mode == "validation":
            # Validation: faster 'vanilla' beamsearch
            bs_nodes = beamsearch_tour_nodes(
                outputs, 1, self.args.batch_size, self.config_tasks[task].num_nodes, probs_type='logits')
        else:
            # Testing: beamsearch with shortest tour heuristic
            bs_nodes = beamsearch_tour_nodes_shortest(
                outputs, inputs["edges_values"], 1, self.args.batch_size, self.config_tasks[task].num_nodes,
                probs_type='logits')

        pred_tour_len, all_tour_lens = mean_tour_len_nodes(inputs["edges_values"], bs_nodes)
        gt_tour_len = torch.mean(inputs["tour_len"])
        opt_gap_sample_wise = (all_tour_lens - inputs["tour_len"]) / inputs["tour_len"]
        opt_gap = opt_gap_sample_wise.mean().item()
        return opt_gap, pred_tour_len, gt_tour_len, opt_gap_sample_wise, bs_nodes

    def reset(self, action="training", dataset_size=1, validation_fine_tune=False):
        if action == "training":
            for task in self.args.train_task_list:
                self.running_data[task] = self.init_running_data()
                self.support_dataset[task] = iter(GoogleTSPReader(self.config_tasks[task].num_nodes,
                                                                  self.config_tasks[task].num_neighbors,
                                                                  self.args.batch_size,
                                                                  self.config_tasks[task].train_filepath,
                                                                  self.config_tasks[task].train_size,
                                                                  split_details={"percentage": dataset_size,
                                                                                 "part": 0},
                                                                  num_batches=self.args.train_steps_per_epoch *
                                                                              self.args.train_finetuning_steps))
        elif action == "validation" and validation_fine_tune:
            for task in self.args.train_task_list:
                self.running_data[task] = self.init_running_data()
                self.support_dataset[task] = iter(GoogleTSPReader(self.config_tasks[task].num_nodes,
                                                                  self.config_tasks[task].num_neighbors,
                                                                  self.args.batch_size,
                                                                  self.config_tasks[task].train_filepath,
                                                                  self.config_tasks[task].train_size,
                                                                  split_details={"percentage": dataset_size,
                                                                                 "part": 0},
                                                                  num_batches=self.args.train_steps_per_epoch *
                                                                              self.args.train_finetuning_steps))

        elif action == "test":
            for task in self.args.test_task_list:
                self.support_dataset[task] = iter(GoogleTSPReader(self.config_tasks[task].num_nodes,
                                                                  self.config_tasks[task].num_neighbors,
                                                                  self.args.batch_size,
                                                                  self.config_tasks[task].train_filepath,
                                                                  num_batches=self.args.test_finetuning_steps,
                                                                  shuffle=False))

    def make_train_logs(self, epoch):
        nb_steps = self.args.train_steps_per_epoch * self.args.train_finetuning_steps
        for task in self.args.train_task_list:
            loss = self.running_data[task]["loss"] / nb_steps
            pred_tour_len = self.running_data[task]["pred_tour_len"] / nb_steps
            gt_tour_len = self.running_data[task]["gt_tour_len"] / nb_steps

            print("=-Task", task, "\tLoss: {0:0.3f}".format(loss),
                  " pred_tour_len: {0:0.3f}".format(pred_tour_len),
                  " gt_tour_len: {0:0.3f}".format(gt_tour_len))

    @staticmethod
    def init_running_data():
        running_data = dict()
        running_data["loss"] = 0.0
        running_data["pred_tour_len"] = 0.0
        running_data["gt_tour_len"] = 0.0
        return running_data

    @staticmethod
    def make_config(data_dir, size, modes=0, scale=0, task_type="reptile"):
        config = dict()
        datafile_prefix = "tsp" + str(size)
        if modes != 0:
            datafile_prefix += "_mode" + str(modes)
        elif scale != 0:
            datafile_prefix += "_scale" + str(scale)
        config["train_filepath"] = os.path.join(data_dir, MODEL, task_type, datafile_prefix + "_train_concorde.txt")
        config["val_filepath"] = os.path.join(data_dir, MODEL, task_type, datafile_prefix + "_val_concorde.txt")
        config["test_filepath"] = os.path.join(data_dir, MODEL, task_type, datafile_prefix + "_test_concorde.txt")
        config["train_size"] = len(open(config["train_filepath"]).readlines())
        config["val_size"] = len(open(config["val_filepath"]).readlines())
        config["test_size"] = len(open(config["test_filepath"]).readlines())
        config["num_nodes"] = size
        config["num_neighbors"] = -1 if size <= 20 else 20
        return DotDict(config)

    def sample_test_tasks(self):
        return self.args.test_task_list

    def sample_train_tasks(self):
        return self.args.train_task_list

    def get_net_params(self):
        return copy.deepcopy(self.net.state_dict())

    def set_net_params(self, parameters):
        self.net.load_state_dict(parameters)

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def optimizer_step(self):
        self.optimizer.step()

    def set_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
            break
        return lr

    def get_config_tasks(self):
        return self.config_tasks
