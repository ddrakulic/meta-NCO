import torch
import os
import time
from supervisors.test_supervisor import run_test


class Supervisor(object):
    def __init__(self, args, task_handler, meta_handler, log_dir, print_avg_metric_test=False):
        self.args = args
        # meta and task handlers
        self.meta_handler = meta_handler
        self.task_handler = task_handler

        # logging
        self.log_dir = log_dir
        self.initialize_loggers()

        self.total_training_time = 0
        self.print_avg_metric_test = print_avg_metric_test

    def train(self, with_validation=True):
        old_val_loss = 1e6
        best_val_metric = 1e6
        epoch = 0
        while True:
            if self.total_training_time / 3600 > self.args.time_limit:
                print("Time limit reached. Aborted after {:.2f} hours".format(self.total_training_time / 3600))
                break
            epoch += 1

            print("=EPOCH", epoch)
            start_time = time.time()

            # prepare data and initialize metrics

            self.task_handler.reset(dataset_size=self.args.support_dataset_size)
            self.meta_handler.reset(self.task_handler.get_config_tasks(), query_size=1-self.args.support_dataset_size)

            for step in range(self.args.train_steps_per_epoch):
                net_meta_old_params = self.meta_handler.get_net_params()

                task_list = self.task_handler.sample_train_tasks()
                # ********** Fine tuning ****************
                for task in task_list:
                    # load meta params to task net
                    self.task_handler.set_net_params(net_meta_old_params)

                    for finetuning_step in range(self.args.train_finetuning_steps):
                        loss = self.task_handler.forward(task)
                        self.task_handler.optimizer_zero_grad()
                        loss.backward()
                        self.task_handler.optimizer_step()

                    # ************ Update meta network **************
                    # save old meta params and load task params to net meta

                    self.meta_handler.set_net_params(self.task_handler.get_net_params())

                    # one step of meta training
                    loss = self.meta_handler.forward(task)
                    loss.backward()

                # load original net_meta params
                self.meta_handler.set_net_params(net_meta_old_params)
                # optimizer_meta step
                self.meta_handler.optimizer_step()
                self.meta_handler.optimizer_zero_grad()

                # =========== END OF TRAINING ============

            training_time = (time.time() - start_time)
            self.total_training_time += training_time
            print("=-Training, time: {0:.2f}min".format(training_time / 60))
            self.task_handler.make_train_logs(epoch)
            self.meta_handler.make_train_logs(epoch)

            # =========== VALIDATION =================
            if with_validation:
                if epoch % self.args.val_every == 0:
                    # keep training net meta params
                    val_loss, val_metric = self.validate(epoch)

                    if val_metric < best_val_metric:
                        best_val_metric = val_metric  # Update best prediction
                        torch.save(self.meta_handler.get_net_params(), self.log_dir + "best_val_checkpoint.tar")

                    # Update learning rate
                    if val_loss > 0.99 * old_val_loss:
                        self.update_learning_rate()

                    old_val_loss = val_loss
            else:
                torch.save(self.meta_handler.get_net_params(), self.log_dir + "best_val_checkpoint.tar")

        # at the end, perform final testing
        run_test(self.task_handler, self.args.test_dataset_size, self.log_dir,
                 self.log_dir + "best_val_checkpoint.tar", epoch=epoch)

    def validate(self, epoch=0):
        sum_val_metric, sum_val_loss = 0., 0.

        start_time = time.time()

        # prepare data and initialize metrics
        self.task_handler.reset(action="validation", validation_fine_tune=True)
        task_list = self.task_handler.sample_train_tasks()

        for task in task_list:
            # load net_meta params
            self.task_handler.set_net_params(self.meta_handler.get_net_params())

            # fine tuning
            for validation_step in range(self.args.train_finetuning_steps):
                # forward pass
                loss = self.task_handler.forward(task, "validation")
                loss.backward()
                self.task_handler.optimizer_step()
                self.task_handler.optimizer_zero_grad()

            val_metric, val_loss, _ = self.task_handler.evaluate(task, "validation")
            sum_val_metric += val_metric
            sum_val_loss += val_loss

        validation_time = (time.time() - start_time)
        self.total_training_time += validation_time
        print("=--Validation, time: {0:.2f}min, sum val metric: {1:.3f}, sum val loss: {2:.3f}".
              format(validation_time / 60, sum_val_metric, sum_val_loss))

        self.write_val_stats(epoch, sum_val_loss, sum_val_metric, self.log_dir)

        return sum_val_loss, sum_val_metric  # Update old validation loss

    def initialize_loggers(self):
        os.makedirs(self.log_dir, exist_ok=True)

    @staticmethod
    def write_val_stats(epoch, val_loss, metric, log_dir):
        with open(os.path.join(log_dir, "val_loss.txt"), "a") as file:
            file.write(str(epoch) + " " + str(val_loss) + "\n")
        with open(os.path.join(log_dir, "val_metric.txt"), "a") as file:
            file.write(str(epoch) + " " + str(metric) + "\n")

    @staticmethod
    def write_test_ft_stats(ft, task, metric, log_dir, epoch=0):
        with open(os.path.join(log_dir, "test_ft_steps-" + str(task) + ".txt"), "a") as file:
            file.write(str(epoch) + " " + str(ft) + " " + str(metric) + "\n")

    def update_learning_rate(self):
        current_learning_rate = self.meta_handler.get_learning_rate()
        new_learning_rate = current_learning_rate * self.args.decay_learning_rate
        self.meta_handler.set_learning_rate(new_learning_rate)
        print("Learning rate uprated to ", new_learning_rate)
