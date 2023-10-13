import torch
from torch import nn
import numpy as np
from base_models.gcn_tsp.models.gcn_model import ResidualGatedGCNModel
from base_models.gcn_tsp.utils.google_tsp_reader import GoogleTSPReader
from base_models.gcn_tsp.utils.graph_utils import mean_tour_len_edges
from supervisors.handlers.base import HandlerBase

MODEL = "gcn_tsp"


class MetaHandler(HandlerBase):
    def __init__(self, config_net, config_train):
        self.config_net = config_net
        self.config_train = config_train
        self.net = nn.DataParallel(ResidualGatedGCNModel(self.config_net))

        if torch.cuda.is_available():
            self.net = self.net.cuda()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config_train.meta_learning_rate)
        self.query_dataset = dict()

        # Print number of network parameters
        nb_param = 0
        for param in self.net.parameters():
            nb_param += np.prod(list(param.data.size()))
        print('Number of parameters:', nb_param)

        self.running_data = dict()

    def reset(self, task_handler):
        self.running_data["loss"] = 0.0
        self.running_data["pred_tour_len"] = 0.0
        self.running_data["gt_tour_len"] = 0.0
        support_dataset_size = self.config_train.finetuning_steps / (1 + self.config_train.finetuning_steps)
        for task in task_handler.train_task_list:
            self.query_dataset[task] = iter(GoogleTSPReader(task_handler.config_tasks[task].num_nodes,
                                                            task_handler.config_tasks[task].num_neighbors,
                                                            self.config_train.batch_size,
                                                            task_handler.config_tasks[task].train_filepath,
                                                            task_handler.config_tasks[task].train_size,
                                                            split_details={"percentage": support_dataset_size,
                                                                           "part": 1},
                                                            num_batches=self.config_train.train_steps_per_epoch))

    def forward(self, task):
        self.net.train()
        inputs = next(self.query_dataset[task])
        outputs, loss = self.net(inputs)
        loss = loss.mean()  # Take mean of loss across multiple GPUs
        # update metrics
        pred_tour_len = mean_tour_len_edges(inputs["edges_values"], outputs)
        gt_tour_len = torch.mean(inputs["tour_len"])
        self.running_data["loss"] += loss.data.item()
        self.running_data["pred_tour_len"] += pred_tour_len
        self.running_data["gt_tour_len"] += gt_tour_len.item()
        return loss

    def make_train_logs(self, epoch, tb_logger):
        nb_steps = self.config_train.train_steps_per_epoch
        print("=-Meta:   \tLoss: {0:0.3f}".format(self.running_data["loss"] / nb_steps),
              " pred_tour_len: {0:0.3f}".format(self.running_data["pred_tour_len"] / nb_steps),
              " gt_tour_len: {0:0.3f}".format(self.running_data["gt_tour_len"] / nb_steps))
        # write loss to tensorboard
        tb_logger.log_value("loss_train", self.running_data["loss"] / nb_steps, epoch)
