import torch
from base_models.gcn_vrp.models.gcn_model_vrp import ResidualGatedGCNModelVRP
import numpy as np
import os
import copy

from base_models.gcn_vrp.models.vrp_reader import VRPReader
from supervisors.config import get_config
from supervisors.handlers.base import TaskHandlerBase

MODEL = "gcn_vrp"


class MultiHandler(TaskHandlerBase):
    def __init__(self, config_net, config_train, working_dir, data_dir, task_type, train_task_list, test_task_list):

        # Instantiate the network
        self.net = ResidualGatedGCNModelVRP(config_net)
        if torch.cuda.is_available():
            self.net = self.net.cuda()

        # Compute number of network parameters
        nb_param = 0
        for param in self.net.parameters():
            nb_param += np.prod(list(param.data.size()))
        print('Number of parameters:', nb_param)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config_train.learning_rate)

        self.config_tasks, self.datasets = dict(), dict()
        self.running_data = None
        self.config_train = config_train

        self.train_task_list = train_task_list
        self.test_task_list = test_task_list

        # load task configs
        for task in set(train_task_list).union(test_task_list):
            self.config_tasks[task] = self.load_config(working_dir, data_dir, task_type, task)
        self.train_datasets = dict()
        print("Loading datasets...")
        for task in train_task_list:
            self.train_datasets[task] = VRPReader(self.config_tasks[task].num_nodes,
                                                  self.config_tasks[task].num_neighbors,
                                                  20,
                                                  self.config_tasks[task].train_filepath)
        print("Datasets loaded.")

    def reset(self, action="training"):
        self.running_data = self.init_running_data()

    def forward(self, task, action="train"):
        self.train_datasets[task].shuffle()
        dataset = iter(self.train_datasets[task])
        inputs = next(dataset)

        outputs, loss = self.net(inputs)
        loss = loss.mean()  # Take mean of loss across multiple GPUs
        if action == "train":
            # update running data
            pred_tour_len = self.mean_tour_len_edges(inputs["edges_values"], outputs)
            gt_tour_len = torch.mean(inputs["tour_len"])
            self.running_data["loss"] += loss.data.item()
            self.running_data["pred_tour_len"] += pred_tour_len
            self.running_data["gt_tour_len"] += gt_tour_len.item()
        return loss

    def evaluate(self, task, action):
        # there is no evaluation in VRP
        return 1e-9, 0, dict()

    def make_train_logs(self, epoch, tb_logger):
        nb_steps = self.config_train.train_steps_per_epoch
        print(", sum_loss: {0:.3f}, sum_pred_tour_len: {1:.3f}, sum_gt_tour_len: {2:.3f}".format(
            self.running_data["loss"] / nb_steps,
            self.running_data["pred_tour_len"] / nb_steps,
            self.running_data["gt_tour_len"] / nb_steps))
        # write loss to tensorboard
        tb_logger.log_value("loss_train", self.running_data["loss"] / nb_steps, epoch)

    @staticmethod
    def load_config(working_dir, data_dir, task_type, task_id, supervisor="multi"):
        filepath = os.path.join(working_dir, "base_models", MODEL, "configs", supervisor, task_type,
                                "config_task-" + str(task_id) + ".json")
        config = get_config(filepath)
        config["train_filepath"] = os.path.join(data_dir, MODEL, task_type, config["train_filename"])
        config["val_filepath"] = os.path.join(data_dir, MODEL, task_type, config["val_filename"])
        config["test_filepath"] = os.path.join(data_dir, MODEL, task_type, config["test_filename"])
        return config

    @staticmethod
    def init_running_data():
        running_data = dict()
        running_data["loss"] = 0.0
        running_data["pred_tour_len"] = 0.0
        running_data["gt_tour_len"] = 0.0
        return running_data

    def set_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_net_params(self):
        return copy.deepcopy(self.net.state_dict())

    def set_net_params(self, parameters):
        self.net.load_state_dict(parameters)

    def sample_test_tasks(self):
        return self.test_task_list

    def sample_train_tasks(self):
        return self.train_task_list

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

    def mean_tour_len_edges(self, x_edges_values, y_pred_edges):
        """
        Computes mean tour length for given batch prediction as edge adjacency matrices (for PyTorch tensors).

        Args:
            x_edges_values: Edge values (distance) matrix (batch_size, num_nodes, num_nodes)
            y_pred_edges: Edge predictions (batch_size, num_nodes, num_nodes, voc_edges)

        Returns:
            mean_tour_len: Mean tour length over batch
        """
        y = torch.nn.functional.softmax(y_pred_edges, dim=-1)  # B x V x V x voc_edges
        y = y.argmax(dim=3)  # B x V x V
        # Divide by 2 because edges_values is symmetric
        tour_lens = (y.float() * x_edges_values.float()).sum(dim=1).sum(dim=1) / 2
        mean_tour_len = tour_lens.sum().to(dtype=torch.float).item() / tour_lens.numel()
        return mean_tour_len