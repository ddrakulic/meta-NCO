"""
    Test is the same for all supervisors
"""

import torch
import os
import time
import numpy as np


def run_test(handler, test_dataset_size, log_dir, model_file, epoch=0, print_avg_metric_test=False):
    # load checkpoint
    if torch.cuda.is_available():
        checkpoint = torch.load(model_file)
    else:
        checkpoint = torch.load(model_file, map_location="cpu")
    handler.set_learning_rate(handler.args.test_finetuning_learning_rate)

    # prepare data and initialize metrics
    handler.reset("test", dataset_size=test_dataset_size)
    task_list = handler.sample_test_tasks()
    sum_metrics = list()
    print("=-Test-=")
    for task in task_list:
        handler.set_net_params(checkpoint)
        metrics = list()
        print("==-Task ", task)

        # fine tuning
        for finetuning_step in range(handler.args.test_finetuning_steps):
            if finetuning_step % handler.args.test_every_n_finetuning_steps == 0:
                # testing after n-th FT step
                start_time = time.time()
                metric, loss, values = handler.evaluate(task, "test", finetuning_step)
                metrics.append(metric)
                print("=--Test after {0} FT steps, time: {1:.2f}min, metric: {2:.3f}, loss: {3:.3f}".
                      format(finetuning_step, (time.time() - start_time) / 60, metric, loss), end="")

                for k, v in values.items():
                    print(", {0}: {1:.3f}".format(k, v), end="")
                print("")
                write_test_ft_stats(finetuning_step, task, metric, log_dir, epoch)
            loss = handler.forward(task, "test")
            handler.optimizer_zero_grad()
            loss.backward()
            metrics.append(loss)
            handler.optimizer_step()

        sum_metrics.append(metrics)

    if print_avg_metric_test:
        avgs = np.mean(np.array(sum_metrics), axis=0)
        for finetuning_step in range(len(avgs)):
            print("=--Test after {0} FT steps, avg metric: {1:.6f}". format(finetuning_step, avgs[finetuning_step]))


def write_test_ft_stats(ft, task, metric, log_dir, epoch=0):
    with open(os.path.join(log_dir, "test_ft_steps-" + str(task) + ".txt"), "a") as file:
        file.write(str(epoch) + " " + str(ft) + " " + str(metric) + "\n")