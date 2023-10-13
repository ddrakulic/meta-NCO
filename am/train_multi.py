import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel
from nets.attention_model import set_decode_type
from utils import move_to

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts, return_all_costs=False):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)

    print("sample costs" , cost[0:20])
    avg_cost = cost.mean()

    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) ))

    if(return_all_costs):
        return avg_cost, cost

    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [

        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(task, model_common, baseline, epoch, val_dataset, problem, tb_logger, opts, test = False):
    # print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    print("task size ", task)
    step = 1#epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    print("epoch = batch ", opts.epoch_size_multi_single)

    training_dataset = baseline.wrap_dataset(problem.make_dataset( num_samples=opts.epoch_size_multi_single, distribution=opts.data_distribution, task=task))

    # Sample from multinomial
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    model_task = model_common

    model_task.train()

    set_decode_type(model_task, "sampling")

    optimizer_update = None

    loss = 0

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        loss+=train_batch(
            model_task,
            optimizer_update,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts, test
        )

        step += 1
    print("loss ", loss)
    loss = loss/(opts.epoch_size_multi_single/opts.batch_size)
    print("loss after divide", loss)

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    loss.backward()


    print(" baseline every 340 epoch ")

    if(epoch%340==0):
        print(" size  ", task)
        avg_reward = validate(model_task, val_dataset, opts)
        print(" Reward After Tune", avg_reward)

        baseline.epoch_callback(model_task, epoch)


    return loss



def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts, test=False
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    return loss