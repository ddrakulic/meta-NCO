#!/usr/bin/env python

import os
import json
import pprint as pp

import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem
from train import clip_grad_norms
import pickle

import datetime
import numpy as np


def run(opts):
    opts.val_dataset = None
    # Pretty print the run args
    pp.pprint(vars(opts))
    val_dict_costs_ins_heu_mean = {}

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    print(" saving in ", opts.save_dir)
    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model_meta = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model_meta)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    tasks_list = []
    if opts.variation_type == 'graph_size':
        graph_sizes = [10, 20, 50]
        if opts.problem == "tsp":
            graph_sizes = [10, 20, 30, 50]

        for g_sizes in graph_sizes:
            task_prop = {'graph_size':g_sizes, 'low':0, 'high':1, 'dist':'uniform', 'variation_type':opts.variation_type}
            task_prop[
                'insertion_heuristic_cost_file'] = "results_all/validation/gsize_{}_valfarthest_insertion.pkl".format(
                task_prop['graph_size'])
            tasks_list.append(task_prop)

    elif opts.variation_type == 'scale':
        scales = [[0,1],[0,2],[0,4]]
        for scale in scales:
            task_prop = {'graph_size': opts.graph_size, 'low': scale[0], 'high': scale[1], 'dist': 'uniform',  'variation_type':opts.variation_type}
            task_prop[
                'insertion_heuristic_cost_file'] = "results_all/validation/SCALE_{}_{}-{}_valfarthest_insertion.pkl".format(
                task_prop['graph_size'], int(task_prop['low']), int(task_prop['high']))
            tasks_list.append(task_prop)

    elif opts.variation_type == 'distribution':
        modes = [1, 2, 5]
        for num_modes in modes:
            task_prop = {'graph_size': opts.graph_size, 'num_modes': num_modes, 'dist': 'gmm',  'variation_type':opts.variation_type}
            task_prop[
                'insertion_heuristic_cost_file'] = "results_all/validation/GRID_{}_modes_{}_valfarthest_insertion.pkl".format(
                 task_prop['graph_size'], task_prop['num_modes'])
            tasks_list.append(task_prop)
    else:
        print(" not matching argument variation type")
        exit()

    baseline_dict = {}
    val_dict = {}
    print("tasks lists ", tasks_list)

    for task in tasks_list:
        model_class = {
            'attention': AttentionModel,
            'pointer': PointerNetwork
        }.get(opts.model, None)

        model = model_class(
            opts.embedding_dim,
            opts.hidden_dim,
            problem,
            n_encode_layers=opts.n_encode_layers,
            mask_inner=True,
            mask_logits=True,
            normalization=opts.normalization,
            tanh_clipping=opts.tanh_clipping,
            checkpoint_encoder=opts.checkpoint_encoder,
            shrink_size=opts.shrink_size
        ).to(opts.device)

        # Overwrite model parameters by parameters to load
        model_ = get_inner_model(model)
        model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

        baseline = RolloutBaseline(model, problem, opts, task=task)
        baseline_dict[str(task)] = baseline

        val_dataset = problem.make_dataset(num_samples=opts.val_size, filename=opts.val_dataset,
            distribution=opts.data_distribution, task=task)
        val_dict[str(task)] = val_dataset

        with open(task['insertion_heuristic_cost_file'], 'rb') as handle:
            insertion_heuristic_costs = pickle.load(handle)

            scale_factor = 1
            if opts.variation_type == 'scale':
                scale_factor = task['high'] - task['low']
                #print("scale factor ", scale_factor)

            insertion_heuristic_costs_mean = np.mean([insertion_heuristic_costs[k]*scale_factor for k in sorted(insertion_heuristic_costs)])
            #print("insertion_heuristic_costs_mean ", insertion_heuristic_costs_mean)

            val_dict_costs_ins_heu_mean[str(task)] = insertion_heuristic_costs_mean

    optimizer_meta = optim.Adam(model_meta.parameters(), opts.lr_model)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer_meta, lambda epoch: opts.lr_decay ** epoch)

    # Main loop
    for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
        now = datetime.datetime.now()
        print("Current date and time : ")
        print(now.strftime("%Y-%m-%d %H:%M:%S"))
        print("we are at epoch id ", epoch)
        #print("tasks list ", len(tasks_list))

        optimizer_meta.zero_grad()
        for index_task, task in enumerate(tasks_list):
            #print("index  task ", index_task,  task)
            baseline = baseline_dict[str(task)]
            val_dataset = val_dict[str(task)]

            # Inner fine-tuning loop + compute task-specific loss on query set
            loss_task, old_weights_meta = train_epoch(task, model_meta,
                                    baseline,
                                    epoch,
                                    val_dataset,
                                    problem,
                                    tb_logger,
                                    opts
                                    )

            #print("loss from task for meta ", loss_task)
            if opts.task_normalization == 1:
                #print("dividing by normalization factor on insertion heuristic ",  val_dict_costs_ins_heu_mean[str(task)])
                loss_task = loss_task / val_dict_costs_ins_heu_mean[str(task)]
                #print("normalized loss task for meta ", loss_task)

            # Task gradient for later meta update
            loss_task.backward()

            # Clip gradient norms and get (clipped) gradient norms for logging
            grad_norms = clip_grad_norms(optimizer_meta.param_groups, opts.max_grad_norm)
            #print("meta loss cur task", loss_task)
            model_meta.load_state_dict(old_weights_meta)

        optimizer_meta.step()
        lr_scheduler.step()
        # print("META_1_2_4 UPDATED")

        if epoch % 15 == 0:
            # print('Saving model and state...')
            torch.save(
                {
                    'model': get_inner_model(model_meta).state_dict(),
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state_all(),
                },
                os.path.join(opts.save_dir, 'epoch_{}_Time{}.pt'.format(epoch, now.strftime("%Y-%m-%d %H:%M:%S")))
            )


if __name__ == "__main__":
    run(get_options())
