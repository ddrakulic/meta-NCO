from base_models.sin_regression.net import SineNet
from supervisors.handlers.base import HandlerBase
import torch
import copy
from torch import nn
MODEL = "sine_regression"


class MetaHandler(HandlerBase):
    def __init__(self, config, task_handler, batch_size=25, k=10):
        self.net = SineNet()
        if torch.cuda.is_available():
            self.net = self.net.cuda()
        self.config = config
        self.batch_size = batch_size
        self.task_handler = task_handler
        self.task_distributions = list()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.meta_learning_rate)
        # num points
        self.k = k
        self.sum_loss = 0

    def forward(self, task, action="train"):
        x, y = list(), list()
        for task in range(self.batch_size):
            x_, y_ = self.task_handler.train_task_distributions[task].sample_data(num_points=self.k)
            x.append(x_)
            y.append(y_)
        x, y = torch.stack(x), torch.stack(y)
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        output = self.net(x)
        loss = self.criterion(output, y)
        if action == "train":
            self.sum_loss += loss
        return loss

    def reset(self, task_handler):
        self.sum_loss = 0
        self.task_distributions = task_handler.train_task_distributions

    def make_train_logs(self, epoch=None, tb_logger=None):
        nb_steps = self.config.train_steps_per_epoch
        print("Meta, sum_loss: {0:.3f}".format(self.sum_loss / nb_steps))

    def set_net_params(self, parameters):
        self.net.load_state_dict(parameters)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.meta_learning_rate)

    def set_learning_rate(self, lr):
        assert hasattr(self, 'optimizer')
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_learning_rate(self):
        assert hasattr(self, 'optimizer')
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def optimizer_step(self):
        assert hasattr(self, 'optimizer')
        self.optimizer.step()

    def optimizer_zero_grad(self):
        assert hasattr(self, 'optimizer')
        self.optimizer.zero_grad()

    def get_net_params(self):
        assert hasattr(self, 'net')
        return copy.deepcopy(self.net.state_dict())
