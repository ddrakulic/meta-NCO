#!/usr/bin/env python

import os
import pprint as pp
#
import torch
from tensorboard_logger import Logger as TbLogger

from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem
from train import clip_grad_norms, tune_and_validate
import re

import pickle
from datetime import datetime

def run(opts):
    tune_sequence_opt_gaps_across_epochs = []
    val_dict = {}
    val_dict_costs_ins_heu  = {}
    min_cost = 1000000
    min_index = 20000000

    dict_val_epoch_vs_best_till_now = {}


    first_epoch_time = None
    for epoch in range(0, 1000000,30 ):
        # Pretty print the run args
        print("epoch id ")
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
        #split_load_path = load_path.spit('_')
        print("load path ", opts.load_path)
        file_match = [x for x in os.listdir(opts.load_path) if re.match('epoch_'+str(epoch)+"_Time*", x)]
        print("load path regex", file_match)

        if(len(file_match) ==0):
            print("processed all epochs")
            break

        print("loaded epoch " ,file_match[0])
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
        
        

        load_path = opts.load_path  + file_match[0]
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

        TEST =True
        model_dict = {}
        baseline_dict = {}

        tasks_list = []
        if (opts.variation_type == 'graph_size'):
            graph_sizes = [10, 20, 50]
            if (opts.problem == "tsp"):
                graph_sizes = [10, 20, 30, 50]


            for g_sizes in graph_sizes:
                task_prop = {'graph_size': g_sizes, 'low': 0, 'high': 1, 'dist': 'uniform',
                             'variation_type': opts.variation_type}

                task_prop['val_dataset'] = "{}{}_{}_seed{}.pkl".format(opts.problem,
                                                                                            task_prop['graph_size'],
                                                                                            "validation", "4321"
                                                                                            )

                task_prop['insertion_heuristic_cost_file'] = "results_all/validation/gsize_{}_valfarthest_insertion.pkl".format(
                                                                                            task_prop['graph_size']
                                                                                            )
                tasks_list.append(task_prop)

        elif (opts.variation_type == 'scale'):

            scales = [[0.0, 1.0], [0.0, 2.0], [0.0, 4.0]]

            for scale in scales:
                task_prop = {'graph_size': opts.graph_size, 'low': scale[0], 'high': scale[1], 'dist': 'uniform',
                             'variation_type': opts.variation_type}

                task_prop['val_dataset'] = "{}__size_{}_scale_{}_{}_{}_seed{}.pkl".format(opts.problem,
                                                                                            task_prop['graph_size'],task_prop['low'],task_prop['high'],
                                                                                            "validation","4321"
                                                                                            )

                task_prop[
                    'insertion_heuristic_cost_file'] = "results_all/validation/SCALE_{}_{}-{}_valfarthest_insertion.pkl".format(
                    task_prop['graph_size'], int(task_prop['low']),int(task_prop['high'])
                )

                tasks_list.append(task_prop)

        elif (opts.variation_type == 'distribution'):
            for i in [1,2,5]:
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

            baseline = RolloutBaseline(model, problem, opts, task=task, update_baseline= False)

            model_dict[str(task)] = model
            baseline_dict[str(task)] = baseline
            print("opts val dataset", opts.val_dataset)


            if(str(task) not in val_dict ):
                print(" val data exists ", task)


                val_dataset = problem.make_dataset(
                     num_samples=opts.val_size, filename=opts.val_dataset+opts.variation_type+'/'+opts.problem+'/'+task['val_dataset'],
                    distribution=opts.data_distribution, task=task)

                val_dict[str(task)] = val_dataset

                with open(task['insertion_heuristic_cost_file'], 'rb') as handle:
                    insertion_heuristic_costs = pickle.load(handle)
                    insertion_heuristic_costs = [insertion_heuristic_costs[k] for k in sorted(insertion_heuristic_costs)]

                    val_dict_costs_ins_heu[str(task)] =insertion_heuristic_costs


            else:
                print(" exists ", task)

        total_opt_gap_across_tasks = 0

        for task in tasks_list:
            baseline = baseline_dict[str(task)]
            val_dataset = val_dict[str(task)]

            opt_gap_wrt_ins_heu_cur_task_updated = tune_and_validate(task, model_meta,
                                    baseline,
                                    epoch,
                                    val_dataset,
                                    problem,
                                    tb_logger,
                                    opts, val_dict_costs_ins_heu[str(task)]
                                    )



            total_opt_gap_across_tasks += opt_gap_wrt_ins_heu_cur_task_updated

        print("EPOCH ID ", epoch)
        avg_opt_gal_val = total_opt_gap_across_tasks/len(tasks_list)
        print("Avg total_opt_gap_across_tasks all tasks after fine tune ", )
        tune_sequence_opt_gaps_across_epochs.append(avg_opt_gal_val)

        for index, x in enumerate(tune_sequence_opt_gaps_across_epochs):
            print(index, x)


        if(min_cost > avg_opt_gal_val ):
            min_cost = avg_opt_gal_val
            min_index = epoch

            print("BEST NEW EPOCH INDEX  NOW ", min_index)
            print("BEST NEW EPOCH COST  NOW ", min_cost)
            dict_val_epoch_vs_best_till_now['OVERALL_BEST_EPOCH_PATH'] = load_path

        print("BEST EPOCH INDEX TILL NOW ", min_index)
        print("BEST EPOCH COST TILL NOW ", min_cost)

        dict_val_epoch_vs_best_till_now[epoch] = {}
        dict_val_epoch_vs_best_till_now[epoch]['best_epoch_till_now'] = min_index
        dict_val_epoch_vs_best_till_now[epoch]['time_of_cur_epoch'] = time_part_of_epoch
        dict_val_epoch_vs_best_till_now[epoch]['opt_gap_wrt_ir_cur_epoch'] = avg_opt_gal_val
        


    print("BEST EPOCH ", min_index)
    print("BEST EPOCH path ", opts.load_path + str(min_index))
    

    print("dict_val_epoch_vs_best_till_now ", dict_val_epoch_vs_best_till_now)
    
    print(" BEST EPOCH LOAD PATH ", dict_val_epoch_vs_best_till_now['OVERALL_BEST_EPOCH_PATH'])

    with open("results_all/validation/META_"+ opts.val_result_pickle_file, 'wb') as handle:
        pickle.dump(dict_val_epoch_vs_best_till_now, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    run(get_options())
