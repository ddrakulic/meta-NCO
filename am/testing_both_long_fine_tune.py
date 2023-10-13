#!/usr/bin/env python

import os
import json
import pprint as pp

import torch
from tensorboard_logger import Logger as TbLogger

from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem
from train import clip_grad_norms, tune_and_test
import pickle

def run(opts):
    tune_sequence = []
    val_dict = {}
    opts.longer_fine_tune = 1

    print("epoch id ")

    with open( opts.val_result_pickle_file, 'rb') as handle:
        dict_val_epoch_vs_best_till_now = pickle.load(handle)
                
            
    epoch =  dict_val_epoch_vs_best_till_now['OVERALL_BEST_EPOCH_PATH']
    opts.load_path = epoch
    
    print(" opts.load_path ", opts.load_path)
    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    # Set the device
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    print("load")
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

    model_dict = {}
    baseline_dict = {}

    tasks_list = []
    if (opts.variation_type == 'graph_size'):

        graph_sizes = [opts.graph_size]
        for g_sizes in graph_sizes:

            task_prop = {'graph_size': g_sizes, 'low': 0, 'high': 1, 'dist': 'uniform',
                         'variation_type': opts.variation_type}

            task_prop['test_dataset'] = "{}{}_{}_seed{}.pkl".format(opts.problem,
                                                                                        task_prop['graph_size'],
                                                                                        "test", "1234"
                                                                                        )


            task_prop['fine_tuning_dataset'] = "{}{}_{}_seed{}.pkl".format(opts.problem,
                                                                                        task_prop['graph_size'],
                                                                                        "fine_tuning", "9999"
                                                                                        )

            tasks_list.append(task_prop)

    elif (opts.variation_type == 'scale'):

        scales = [[0.0,opts.scale*1.0]]


        for scale in scales:
            task_prop = {'graph_size': opts.graph_size, 'low': scale[0], 'high': scale[1], 'dist': 'uniform',
                         'variation_type': opts.variation_type}

            task_prop['test_dataset'] = "{}__size_{}_scale_{}_{}_{}_seed{}.pkl".format(opts.problem,
                                                                                        task_prop['graph_size'],task_prop['low'],task_prop['high'],
                                                                                        "test","1234"
                                                                                        )

            task_prop['fine_tuning_dataset'] = "{}__size_{}_scale_{}_{}_{}_seed{}.pkl".format(opts.problem,
                                                                                       task_prop['graph_size'],
                                                                                       task_prop['low'],
                                                                                       task_prop['high'],
                                                                                       "fine_tuning", "9999"
                                                                                       )

            tasks_list.append(task_prop)

    elif (opts.variation_type == 'distribution'):

        for i in [opts.num_modes]:
            num_modes = i

            task_prop = {'graph_size': opts.graph_size, 'num_modes': num_modes, 'dist': 'gmm',
                         'variation_type': opts.variation_type}

            task_prop['test_dataset'] = "{}__size_{}_distribution_{}_{}_seed{}.pkl".format(opts.problem,
                                                                                          task_prop['graph_size'],
                                                                                          str(num_modes),
                                                                                          "test", "1234"
                                                                                          )

            task_prop['fine_tuning_dataset'] = "{}__size_{}_distribution_{}_{}_seed{}.pkl".format(opts.problem,
                                                                                           task_prop['graph_size'],
                                                                                           str(num_modes),
                                                                                           "fine_tuning", "9999"
                                                                                           )


            tasks_list.append(task_prop)

    else:
        print(" not matching argument variation type")
        exit()

    for task in tasks_list:

        print("task ", task)
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

        baseline = RolloutBaseline(model, problem, opts, task=task, update_baseline=False)

        model_dict[str(task)] = model
        baseline_dict[str(task)] = baseline
        print("opts val dataset", opts.val_dataset)


        if(str(task) not in val_dict ):
            print(" val data exists ", task)


            val_dataset = problem.make_dataset(
                 num_samples=opts.test_size, filename=opts.val_dataset+opts.variation_type+'/'+opts.problem+'/'+task['test_dataset'],
                distribution=opts.data_distribution, task=task)

            val_dict[str(task)] = val_dataset

        else:
            print(" exists ", task)

    total_reward_tasks = 0

    dict_results_task_sample_iter_wise = {}

    for task in tasks_list:

        task_string = None
        if(opts.variation_type =='distribution'):
            task_string = task['num_modes']

        if (opts.variation_type == 'scale'):
            task_string = str(task['low']) +'_' + str(task['high'])

        if (opts.variation_type == 'graph_size'):
            task_string = task['graph_size']


        baseline = baseline_dict[str(task)]
        val_dataset = val_dict[str(task)]


        dict_results_task_sample_iter_wise[task_string] = {}


        fine_tuning_dataset = problem.make_dataset(
            filename=opts.val_dataset + opts.variation_type + '/' + opts.problem + '/' + task['fine_tuning_dataset'], task=task)

        updated_reward = tune_and_test(task, model_meta,
                                baseline,
                                epoch,
                                val_dataset,
                                problem,
                                tb_logger,
                                opts, fine_tuning_dataset, dict_results_task_sample_iter_wise[task_string]
                                )

        total_reward_tasks+=updated_reward

    with open("results_all/test/TEST_LONG_" +opts.test_result_pickle_file, 'wb') as handle:
        pickle.dump(dict_results_task_sample_iter_wise, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("EPOCH ID ", opts.load_path)
    avg_rewards_val = total_reward_tasks/len(tasks_list)
    print("Avg reward all tasks after fine tune ", )
    tune_sequence.append(avg_rewards_val)

    for index, x in enumerate(tune_sequence):
        print(index, x.data)


    print(" dict_results_task_sample_iter_wise ", dict_results_task_sample_iter_wise)


if __name__ == "__main__":
    run(get_options())
