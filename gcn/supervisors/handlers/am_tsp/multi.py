import torch
import numpy as np
from base_models.am_tsp.nets.attention_model import AttentionModel
from supervisors.handlers.task_handler import TaskHandlerBase

MODEL = "am_tsp"


class MultiHandler(TaskHandlerBase):

    def __init__(self, opts):
        # Model
        self.net = AttentionModel(
            opts.embedding_dim,
            opts.hidden_dim,
            "tsp",
            n_encode_layers=opts.n_encode_layers,
            mask_inner=True,
            mask_logits=True,
            normalization=opts.normalization,
            tanh_clipping=opts.tanh_clipping,
            checkpoint_encoder=opts.checkpoint_encoder,
            shrink_size=opts.shrink_size
        )
        if torch.cuda.is_available():
            self.net = self.net.cuda()

        # Number of network parameters
        nb_param = 0
        for param in self.net.parameters():
            nb_param += np.prod(list(param.data.size()))
        print('Number of parameters:', nb_param)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opts.lr_model)

        self.train_task_list = opts.train_task_list
        self.test_task_list = opts.test_task_list

    def reset(self, action="training"):
        raise NotImplementedError

    def forward(self, task, action="train"):
        raise NotImplementedError

    def evaluate(self, task, action):
        raise NotImplementedError

    def make_train_logs(self, epoch, tb_logger):
        raise NotImplementedError

    def sample_train_tasks(self):
        raise NotImplementedError

    def sample_test_tasks(self):
        raise NotImplementedError
