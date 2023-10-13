from base_models.sin_regression.net import SineNet, SineDistribution
from supervisors.handlers.base import TaskHandlerBase
import copy
import torch
from torch import nn
MODEL = "sine_regression"


class TaskHandler(TaskHandlerBase):
    def __init__(self, config, min_amp, max_amp, min_phase, max_phase, min_x, max_x, k=10,
                 batch_size=25, num_testtasks=2):
        self.net = SineNet()
        if torch.cuda.is_available():
            self.net = self.net.cuda()
        self.task_distribution = SineDistribution(min_amp, max_amp, min_phase, max_phase, min_x, max_x)
        self.config = config
        self.train_task_distributions = list()
        self.test_task_distribution = list()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=config.task_learning_rate)
        # num points
        self.k = k
        self.num_testtasks = num_testtasks
        self.sum_loss = 0
        self.test_data = list()
        self.batch_size = batch_size

    def forward(self, task, action="train"):
        if action == "train":
            x, y = list(), list()
            for task in range(self.batch_size):
                x_, y_ = self.train_task_distributions[task].sample_data(num_points=self.k)
                x.append(x_)
                y.append(y_)
            x, y = torch.stack(x), torch.stack(y)
        elif action == "test":
            x, y = list(), list()
            for task in range(self.batch_size):
                x_, y_ = self.task_distribution.sample_task().sample_data(num_points=self.k)
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

    def reset(self, action="training"):
        if action == "training":
            self.sum_loss = 0
        self.train_task_distributions = list()

    def make_train_logs(self, epoch=None, tb_logger=None):
        nb_steps = self.config.train_steps_per_epoch * self.config.finetuning_steps
        print("Task, sum_loss: {0:.3f}".format(self.sum_loss / nb_steps))

    def sample_train_tasks(self):
        # sample tasks
        self.train_task_distributions = list()
        for _ in range(self.batch_size):
            self.train_task_distributions.append(self.task_distribution.sample_task())
        return range(2)

    def sample_test_tasks(self):
        return range(2)

    def evaluate(self, task, action):
        x, y = self.test_data[task]
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        output = self.net(x)
        loss = self.criterion(output, y)
        return loss.item(), loss.item(), {}

    def set_net_params(self, parameters):
        self.net.load_state_dict(parameters)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.config.task_learning_rate)

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
