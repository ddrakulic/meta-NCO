#!/usr/bin/env python

import os
import json
import pprint as pp

import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from options import get_options
from train_multi import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem
from train_multi import clip_grad_norms
import datetime

def run(opts):

    # Pretty print the run args
    opts.val_dataset = None
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:1" if opts.use_cuda else "cpu")

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
    model_common = model_class(
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
    model_ = get_inner_model(model_common)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    TEST =False

    tasks_list = []
    if (opts.variation_type == 'graph_size'):
        # graph_sizes = [10, 20, 50]
        if (opts.problem == "tsp"):
            graph_sizes = [opts.graph_size]
        for g_sizes in graph_sizes:
            task_prop = {'graph_size': g_sizes, 'low': 0, 'high': 1, 'dist': 'uniform',
                         'variation_type': opts.variation_type}
            tasks_list.append(task_prop)

    elif (opts.variation_type == 'scale'):

        
        scales = [[0,opts.scale]]
        for scale in scales:
            print(scale)
            task_prop = {'graph_size': opts.graph_size, 'low': scale[0], 'high': scale[1], 'dist': 'uniform',
                         'variation_type': opts.variation_type}
            tasks_list.append(task_prop)


    elif (opts.variation_type == 'distribution'):

        print("num modes ", opts.num_modes)
        for i in [opts.num_modes]:
            num_modes = i
            task_prop = {'graph_size': opts.graph_size, 'num_modes': num_modes, 'dist': 'gmm',
                         'variation_type': opts.variation_type}
            tasks_list.append(task_prop)

    else:
        print(" not matching argument variation type")
        exit()

    model_dict = {}
    baseline_dict = {}
    val_dict = {}

    for task in tasks_list:


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

        model_dict[str(task)] = model
        baseline_dict[str(task)] = baseline

        val_dataset = problem.make_dataset( num_samples=opts.val_size, filename=opts.val_dataset,
            distribution=opts.data_distribution, task=task)

        val_dict[str(task)] = val_dataset

    optimizer_common = optim.Adam( model_common.parameters(), opts.lr_model )
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer_common, lambda epoch: opts.lr_decay ** epoch)

    if(TEST ==False):
            print("Training")

            for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
                now = datetime.datetime.now()
                print("Current date and time : ")
                print(now.strftime("%Y-%m-%d %H:%M:%S"))
                print("we are at epoch id ", epoch)

                optimizer_common.zero_grad()
                for task in tasks_list:
                    baseline = baseline_dict[str(task)]
                    val_dataset = val_dict[str(task)]

                    loss_task = train_epoch(task, model_common,
                                            baseline,
                                            epoch,
                                            val_dataset,
                                            problem,
                                            tb_logger,
                                            opts
                                            )
                    print("loss task ", loss_task)

                    # Clip gradient norms and get (clipped) gradient norms for logging
                    grad_norms = clip_grad_norms(optimizer_common.param_groups, opts.max_grad_norm)
                    print("common loss cur task", loss_task)


                optimizer_common.step()

                lr_scheduler.step()
                print("Common  UPDATED")

                if (epoch % 500 == 0):
                    print('Saving model and state...')
                    torch.save(
                        {
                            'model': get_inner_model(model_common).state_dict(),
                            'rng_state': torch.get_rng_state(),
                            'cuda_rng_state': torch.cuda.get_rng_state_all(),
                        },
                        os.path.join(opts.save_dir, 'epoch_{}_Time{}.pt'.format(epoch, now.strftime("%Y-%m-%d %H:%M:%S")))
                    )


if __name__ == "__main__":
    run(get_options())
