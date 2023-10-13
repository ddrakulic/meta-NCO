import copy
import torch
import os
import numpy as np
from base_models.gcn_vrp.models.vrp_reader import VRPReader
from base_models.gcn_vrp.models.gcn_model_vrp import ResidualGatedGCNModelVRP
from base_models.gcn_tsp.utils.graph_utils import mean_tour_len_edges
from supervisors.handlers.base import TaskHandlerBase

MODEL = "gcn_vrp"


class TaskHandler(TaskHandlerBase):
    def __init__(self, config_net, config_train, working_dir, data_dir, task_type, train_task_list, test_task_list):
        self.config_net = config_net
        self.config_train = config_train
        self.net = ResidualGatedGCNModelVRP(self.config_net)

        if torch.cuda.is_available():
            self.net = self.net.cuda()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config_train.task_learning_rate)

        # load task_configs
        self.config_tasks, self.running_data, self.datasets = dict(), dict(), dict()

        self.train_task_list = train_task_list
        self.test_task_list = test_task_list

        for task in set(train_task_list).union(test_task_list):
            self.config_tasks[task] = self.load_config(working_dir, data_dir, task_type, task)

        self.train_datasets, self.test_datasets = dict(), dict()
        print("Loading datasets...")
        for task in train_task_list:
            self.train_datasets[task] = VRPReader(self.config_tasks[task].num_nodes,
                                                  self.config_tasks[task].num_neighbors,
                                                  20,
                                                  self.config_tasks[task].test_filepath)
        for task in test_task_list:
            self.test_datasets[task] = VRPReader(self.config_tasks[task].num_nodes,
                                                 self.config_tasks[task].num_neighbors,
                                                 20,
                                                 self.config_tasks[task].test_filepath)
        print("Datasets loaded.")

    def forward(self, task, action="train"):
        # forward pass
        self.net.train()

        self.train_datasets[task].shuffle()
        dataset = iter(self.train_datasets[task])
        inputs = next(dataset)

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

    def evaluate(self, task, action):
        if action == "test":
            dataset = iter(self.test_datasets[task])
            all_losses = list()
            for iteration in range(200):
                inputs = next(dataset)
                _, loss = self.net(inputs)
                all_losses.append(loss.item())

            return 1e-9, np.mean(all_losses), dict()
        else:
            return 1e-9, 0, dict()

    def reset(self, action="training"):
        for task in self.train_task_list:
            self.running_data[task] = self.init_running_data()

    def make_train_logs(self, epoch, tb_logger):
        nb_steps = self.config_train.train_steps_per_epoch * self.config_train.finetuning_steps
        for task in self.train_task_list:
            loss = self.running_data[task]["loss"] / nb_steps
            pred_tour_len = self.running_data[task]["pred_tour_len"] / nb_steps
            gt_tour_len = self.running_data[task]["gt_tour_len"] / nb_steps

            print("=-Task", task, "\tLoss: {0:0.3f}".format(loss),
                  " pred_tour_len: {0:0.3f}".format(pred_tour_len),
                  " gt_tour_len: {0:0.3f}".format(gt_tour_len))
            # write loss to tensorboard
            tb_logger.log_value("loss_train_task" + str(task), loss, epoch)

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
            datafile_prefix += "_modes" + str(modes)
        elif scale != 0:
            datafile_prefix += "_scale" + str(scale)
        config["train_filepath"] = os.path.join(data_dir, MODEL, task_type, datafile_prefix + "_train_concorde.txt")
        config["val_filepath"] = os.path.join(data_dir, MODEL, task_type, datafile_prefix + "_val_concorde.txt")
        config["test_filepath"] = os.path.join(data_dir, MODEL, task_type, datafile_prefix + "_test_concorde.txt")
        config["train_size"] = sum(1 for _ in open(config["train_filepath"]))
        config["val_size"] = sum(1 for _ in open(config["val_filepath"]))
        config["test_size"] = sum(1 for _ in open(config["test_filepath"]))
        config["num_nodes"] = size
        config["num_neighbors"] = -1 if size > 20 else 20
        return config

    def sample_test_tasks(self):
        return self.test_task_list

    def sample_train_tasks(self):
        return self.train_task_list

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
