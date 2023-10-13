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
import pickle
import re

from datetime import datetime
import pathlib

def run(opts):
    val_dict = {}
    val_dict_costs_ins_heu = {}
    min_cost =100000
    min_index = 1000000
    epoch_vs_performance = []
    dict_val_epoch_vs_best_till_now = {}
    first_epoch_time = None

    for epoch in range(0, 1061000,1500):
        # Pretty print the run args
        print("epoch id ")
        pp.pprint(vars(opts))

        # Set the random seed
        torch.manual_seed(opts.seed)

        # Optionally configure tensorboard
        tb_logger = None
        if not opts.no_tensorboard:
            tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

        # Set the device
        opts.device = torch.device("cuda" if opts.use_cuda else "cpu")
        torch.rand(10).to(opts.device)

        # Figure out what's the problem
        problem = load_problem(opts.problem)

        # Load data from load_path
        load_data = {}
        assert opts.load_path is not None

        print("load path ", opts.load_path)
        file_match = [x for x in os.listdir(opts.load_path) if re.match('epoch_' + str(epoch) + "_Time*", x)]
        print("load path regex", file_match)

        if (len(file_match) == 0):
            print("processed all epochs")
            break

        print("loaded epoch ", file_match[0])
        time_part_of_epoch = re.search('Time(.+?).pt', file_match[0]).group(1)
        print("epoch was saved at time ", time_part_of_epoch)
        
        
        datetime_cur_epoch = datetime.strptime(time_part_of_epoch, '%Y-%m-%d %H:%M:%S')
        
        if(first_epoch_time is None):
            first_epoch_time = datetime_cur_epoch
            
        
        time_diff = datetime_cur_epoch-first_epoch_time
        print("time diff", time_diff)
        time_diff_minutes = int((time_diff.total_seconds())/60)
        print("time diff  minutes since first epoch ", time_diff_minutes )
        if(time_diff_minutes > 1440):
            print(" 24 hours done")
            break

        load_path = opts.load_path + file_match[0]
        print(load_path)

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

        tasks_list = []
        if (opts.variation_type == 'graph_size'):
            graph_sizes = [10, 20, 50]
            if (opts.problem == "tsp"):
                graph_sizes = [opts.graph_size]

            for g_sizes in graph_sizes:
                task_prop = {'graph_size': g_sizes, 'low': 0, 'high': 1, 'dist': 'uniform',
                             'variation_type': opts.variation_type}

                task_prop['val_dataset'] = "{}{}_{}_seed{}.pkl".format(opts.problem,
                                                                       task_prop['graph_size'],
                                                                       "validation", "4321"
                                                                       )

                task_prop[
                    'insertion_heuristic_cost_file'] = "results_all/validation/gsize_{}_valfarthest_insertion.pkl".format(
                    task_prop['graph_size']
                )

                tasks_list.append(task_prop)

        elif (opts.variation_type == 'scale'):

            scales = [[0,opts.scale]]
            for scale in scales:
                task_prop = {'graph_size': opts.graph_size, 'low': scale[0]*1.0, 'high': scale[1]*1.0, 'dist': 'uniform',
                             'variation_type': opts.variation_type}

                task_prop['val_dataset'] = "{}__size_{}_scale_{}_{}_{}_seed{}.pkl".format(opts.problem,
                                                                                          task_prop['graph_size'],
                                                                                          task_prop['low'],
                                                                                          task_prop['high'],
                                                                                          "validation", "4321"
                                                                                          )
                task_prop[
                    'insertion_heuristic_cost_file'] = "results_all/validation/SCALE_{}_{}-{}_valfarthest_insertion.pkl".format(
                    task_prop['graph_size'], int(task_prop['low']), int(task_prop['high'])
                )

                tasks_list.append(task_prop)

        elif (opts.variation_type == 'distribution'):
            for i in [opts.num_modes
                      ]:
                num_modes = i

                task_prop = {'graph_size': opts.graph_size, 'num_modes': num_modes, 'dist': 'gmm',
                             'variation_type': opts.variation_type}

                task_prop['val_dataset'] = "{}__size_{}_distribution_{}_{}_seed{}.pkl".format(opts.problem,
                                                                                              task_prop['graph_size'],
                                                                                              str(num_modes),
                                                                                              "validation", "4321"
                                                                                              )

                task_prop[
                    'insertion_heuristic_cost_file'] = "results_all/validation/GRID_{}_modes_{}_valfarthest_insertion.pkl".format(
                    task_prop['graph_size'], task_prop['num_modes']
                )

                tasks_list.append(task_prop)

        else:
            print(" not matching argument variation type")
            exit()


        model_dict = {}
        baseline_dict = {}

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

            baseline = RolloutBaseline(model, problem, opts, task=task, update_baseline=False)

            model_dict[str(task)] = model
            baseline_dict[str(task)] = baseline

            if(str(task) not in val_dict):
                print("dataset does not exists ", task)
                val_dataset = problem.make_dataset( num_samples=opts.val_size, filename=opts.val_dataset+opts.variation_type+'/'+opts.problem+'/'+task['val_dataset'],
                    distribution=opts.data_distribution, task=task)

                val_dict[str(task)] = val_dataset


                with open(task['insertion_heuristic_cost_file'], 'rb') as handle:
                    insertion_heuristic_costs = pickle.load(handle)
                    insertion_heuristic_costs = [insertion_heuristic_costs[k] for k in sorted(insertion_heuristic_costs)]

                    val_dict_costs_ins_heu[str(task)] =insertion_heuristic_costs

            else :
                print(task," exists ")

        optimizer_common = optim.Adam( model_common.parameters(), opts.lr_model )

        # Initialize learning rate scheduler, decay by lr_decay once per epoch!
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer_common, lambda epoch: opts.lr_decay ** epoch)


        total_opt_gap = 0
        scale_multiply = 1
        if (opts.variation_type == 'scale'):
            scale_multiply = task['high'] - task['low']
        # optimizer_common.zero_grad()
        for task in tasks_list:
            scale_multiply = 1
            if (opts.variation_type == 'scale'):
                scale_multiply = task['high'] - task['low']

            baseline = baseline_dict[str(task)]
            val_dataset = val_dict[str(task)]

            reward_task,all_costs = validate(model_common, val_dataset, opts, return_all_costs=True)

            opt_gap_wrt_ins_heu_cur_task = 0
            for multi_model_cost, ins_cost in zip(all_costs, val_dict_costs_ins_heu[str(task)]):
                # print(before_fine_tune_cost, ins_cost)
                opt_gap_wrt_ins_heu_cur_task += (multi_model_cost.item() - (ins_cost*scale_multiply) ) * 100.0 / ((ins_cost*scale_multiply))

            opt_gap_wrt_ins_heu_cur_task = opt_gap_wrt_ins_heu_cur_task / len(val_dict_costs_ins_heu[str(task)])

            print("task ", task, " reward ", reward_task)
            print("task ", task, " opt_gap_wrt_ins_heu_cur_task ", opt_gap_wrt_ins_heu_cur_task)

            total_opt_gap += opt_gap_wrt_ins_heu_cur_task

        avg_opt_gal_val = total_opt_gap/(len(tasks_list)) # 1 in this case

        print("Avg total_opt_gap_across_tasks all tasks at epoch  ", epoch, avg_opt_gal_val)
        epoch_vs_performance.append(avg_opt_gal_val)

        for index, perf in enumerate(epoch_vs_performance):
            print(index, perf)


        if(min_cost> avg_opt_gal_val):
            min_cost = avg_opt_gal_val
            min_index = epoch

            print("NEW BEST EPOCH INDEX  NOW ", min_index)
            print("NEW BEST EPOCH COST TILL NOW ", min_cost)
            dict_val_epoch_vs_best_till_now['OVERALL_BEST_EPOCH_PATH'] = load_path


        print("BEST EPOCH ", min_index)
        print("BEST COST ", min_cost)

        dict_val_epoch_vs_best_till_now[epoch] = {}
        dict_val_epoch_vs_best_till_now[epoch]['best_epoch_till_now'] = min_index
        dict_val_epoch_vs_best_till_now[epoch]['time_of_cur_epoch'] = time_part_of_epoch
        dict_val_epoch_vs_best_till_now[epoch]['opt_gap_wrt_ir_cur_epoch'] = avg_opt_gal_val

    print("BEST EPOCH path ", opts.load_path+str(min_index))
    
    print(" BEST EPOCH LOAD PATH ", dict_val_epoch_vs_best_till_now['OVERALL_BEST_EPOCH_PATH'])

    print("dict_val_epoch_vs_best_till_now ", dict_val_epoch_vs_best_till_now)

    root_folder_name = ''
    if (opts.variation_type == 'graph_size'):
        root_folder_name = 'SIZE'
    if (opts.variation_type == 'distribution'):
        root_folder_name = 'MODE'
    if (opts.variation_type == 'scale'):
        root_folder_name = 'SCALE'

    val_save_dir = "results_all/validation/" + root_folder_name + '/'

    pathlib.Path(val_save_dir).mkdir(parents=True, exist_ok=True)

    with open(val_save_dir + opts.val_result_pickle_file, 'wb') as handle:
        pickle.dump(dict_val_epoch_vs_best_till_now, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    run(get_options())
