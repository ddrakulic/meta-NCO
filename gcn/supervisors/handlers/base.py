from abc import abstractmethod
import copy


class HandlerBase:
    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, task):
        raise NotImplementedError

    @abstractmethod
    def make_train_logs(self, epoch, tb_logger):
        raise NotImplementedError

    @abstractmethod
    def set_learning_rate(self, lr):
        raise NotImplementedError

    @abstractmethod
    def get_learning_rate(self):
        raise NotImplementedError

    @abstractmethod
    def optimizer_step(self):
        raise NotImplementedError

    @abstractmethod
    def optimizer_zero_grad(self):
        raise NotImplementedError

    @abstractmethod
    def get_net_params(self):
        raise NotImplementedError

    @abstractmethod
    def set_net_params(self, parameters):
        raise NotImplementedError


class TaskHandlerBase(HandlerBase):

    @abstractmethod
    def evaluate(self, task, action):
        raise NotImplementedError

    @abstractmethod
    def get_config_net(self):
        raise NotImplementedError

