import torch
from torch import nn
from base_models.gcn_tsp.models.gcn_model import ResidualGatedGCNModel
from base_models.gcn_tsp.utils.google_tsp_reader import GoogleTSPReader
from base_models.gcn_tsp.utils.graph_utils import mean_tour_len_edges
from supervisors.handlers.base import HandlerBase
import copy

MODEL = "gcn_tsp"


class MetaHandler(HandlerBase):
    def __init__(self, args, config_net):
        self.config_net = config_net
        self.args = args
        self.net = nn.DataParallel(ResidualGatedGCNModel(self.config_net))

        if torch.cuda.is_available():
            self.net = self.net.cuda()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.meta_learning_rate)
        self.query_dataset = dict()
        self.running_data = dict()

    def reset(self, task_configs, query_size):
        self.running_data["loss"] = 0.0
        self.running_data["pred_tour_len"] = 0.0
        self.running_data["gt_tour_len"] = 0.0
        for task in self.args.train_task_list:
            self.query_dataset[task] = iter(GoogleTSPReader(task_configs[task].num_nodes,
                                                            task_configs[task].num_neighbors,
                                                            self.args.batch_size,
                                                            task_configs[task].train_filepath,
                                                            task_configs[task].train_size,
                                                            split_details={"percentage": 1-query_size,
                                                                           "part": 1},
                                                            num_batches=self.args.train_steps_per_epoch))

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

    def make_train_logs(self, epoch):
        nb_steps = self.args.train_steps_per_epoch
        print("=-Meta:   \tLoss: {0:0.3f}".format(self.running_data["loss"] / nb_steps),
              " pred_tour_len: {0:0.3f}".format(self.running_data["pred_tour_len"] / nb_steps),
              " gt_tour_len: {0:0.3f}".format(self.running_data["gt_tour_len"] / nb_steps))

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
