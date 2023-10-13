import torch
import os
import time
from supervisors.test_supervisor import run_test


class Supervisor(object):
    def __init__(self, args, handler, log_dir, print_avg_metric_test=False):
        # handler
        self.handler = handler
        # training hyperparameters
        self.args = args

        # logging
        self.log_dir = log_dir
        self.initialize_loggers()

        self.total_training_time = 0
        self.print_avg_metric_test = print_avg_metric_test

    def train(self, with_validation=True):
        old_val_loss = 1e6
        best_val_metric = 1e6
        alpha = self.args.alpha
        epoch = 0
        while True:
            # infinite loop, will be terminated when time limit is reached
            if self.total_training_time / 3600 > self.args.time_limit:
                print("Time limit reached. Aborted after {:.2f} hours".format(self.total_training_time / 3600))
                break
            epoch += 1

            print("=EPOCH", epoch)
            start_time = time.time()

            # prepare data and initialize metrics
            self.handler.reset()

            task_list = self.handler.sample_train_tasks()
            # ********** Fine tuning ****************
            for task in task_list:
                current_params = self.handler.get_net_params()

                for finetuning_step in range(self.args.train_finetuning_steps):
                    loss = self.handler.forward(task)
                    self.handler.optimizer_zero_grad()
                    loss.backward()
                    self.handler.optimizer_step()

                candidate_params = self.handler.get_net_params()

                # ************ Update network parameters current + alpha * (candidate - current) **************
                state_dict = {param: (current_params[param] + alpha *
                                      (candidate_params[param] - current_params[param]))
                              for param in candidate_params.keys()}
                self.handler.set_net_params(state_dict)

            # =========== END OF TRAINING ============

            training_time = (time.time() - start_time)
            self.total_training_time += training_time
            print("=-Training, time: {0:.2f}min".format(training_time / 60))
            self.handler.make_train_logs(epoch)

            # update alpha
            alpha = alpha * self.args.decay_alpha
            print(" Alpha is updated, new value: ", alpha)

            # =========== VALIDATION =================
            if with_validation:
                if epoch % self.args.val_every == 0:
                    val_loss, val_metric = self.validate(epoch)

                    if val_metric <= best_val_metric:
                        best_val_metric = val_metric  # Update best prediction
                        torch.save(self.handler.get_net_params(), self.log_dir + "best_val_checkpoint.tar")

                    # Update learning rate
                    if val_loss > 0.99 * old_val_loss:
                        self.update_learning_rate()

                    old_val_loss = val_loss
            else:
                # for simple models, we do not perform validation, saving model in every epoch
                torch.save(self.handler.get_net_params(), self.log_dir + "best_val_checkpoint.tar")

        # at the end, perform final testing
        run_test(self.handler, self.args.test_dataset_size, self.log_dir,
                 self.log_dir + "best_val_checkpoint.tar", epoch=epoch)

    def validate(self, epoch=0):
        sum_val_metric, sum_val_loss = 0., 0.
        start_time = time.time()

        # prepare data and initialize metrics
        self.handler.reset("validation")
        task_list = self.handler.sample_train_tasks()
        for task in task_list:
            # load net_meta params
            val_metric, val_loss, _ = self.handler.evaluate(task, "validation")
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

    def update_learning_rate(self):
        current_learning_rate = self.handler.get_learning_rate()
        new_learning_rate = current_learning_rate * self.args.decay_learning_rate
        self.handler.set_learning_rate(new_learning_rate)
        print("Learning rate uprated to ", new_learning_rate)
