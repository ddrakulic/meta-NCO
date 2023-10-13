from base_models.sin_regression.net import SineNet, SineDistribution
from supervisors.handlers.base import TaskHandlerBase
import torch
from torch import nn
import numpy as np
MODEL = "sine_regression"


class MultiHandler(TaskHandlerBase):
    def __init__(self, config, min_amp, max_amp, min_phase, max_phase, min_x, max_x, k=10, num_metatasks=1000,
                 num_testtasks=50):
        self.net = SineNet()
        if torch.cuda.is_available():
            self.net = self.net.cuda()
        self.task_distribution = SineDistribution(min_amp, max_amp, min_phase, max_phase, min_x, max_x)
        self.config = config
        self.train_tasks = dict()
        self.test_task_distribution = dict()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.learning_rate)
        # num points
        self.k = k
        self.num_metatasks = num_metatasks
        self.num_testtasks = num_testtasks
        self.sum_loss = 0
        self.test_data = dict()
        np.random.seed(1)
        for task in range(self.num_testtasks):
            task_distribution = self.task_distribution.sample_task()
            self.test_task_distribution[task] = task_distribution
            self.test_data[task] = task_distribution.sample_data(size=self.k)
        np.random.seed(0)

    def forward(self, task, action="train"):
        if action == "train":
            x, y = self.train_tasks[task].sample_data(size=self.k)
        elif action == "test":
            x, y = self.test_task_distribution[task].sample_data(size=self.k)

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        output = self.net(x)
        loss = self.criterion(output, y) / self.k
        if action == "train":
            self.sum_loss += loss
        return loss

    def reset(self, action="training"):
        if action == "training":
            self.sum_loss = 0
            for task in range(self.num_metatasks):
                self.train_tasks[task] = self.task_distribution.sample_task()

    def make_train_logs(self, epoch=None, tb_logger=None):
        nb_steps = self.num_metatasks
        print(", sum_loss: {0:.3f}".format(self.sum_loss / nb_steps))

    def sample_train_tasks(self):
        return range(self.num_metatasks)

    def sample_test_tasks(self):
        return range(self.num_testtasks)

    def evaluate(self, task, action):
        x, y = self.test_data[task]
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        output = self.net(x)
        loss = self.criterion(output, y) / self.k
        return loss.item(), loss.item(), {}
