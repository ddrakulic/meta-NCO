from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.tsp.state_tsp import StateTSP
from utils.beam_search import beam_search
# from sklearn.datasets.samples_generator import make_blobs
import scipy.stats
import random
import numpy as np
import math
from numpy.random import default_rng
from numpy import meshgrid, array, random

class TSP(object):
    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
                torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
                pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class TSPDataset(Dataset):

    def __init__(self, filename=None,num_samples=1000000, offset=0, distribution=None, task=None):
        super(TSPDataset, self).__init__()
        # print("task ", task)
        # print("inside tsp data set loader ", task['graph_size'])

        self.data_set = []
        if filename is not None:

            filename = filename.replace('XXX', str(task['graph_size']))
            # print("file name ", filename)
            assert os.path.splitext(filename)[1] == '.pkl'

            # print("loading dataset")

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset + num_samples])]
                
                # for scaling to 0-3 since trained mmodel was on 0-3
                if('rescale_for_testing' in task and task['rescale_for_testing'] is not None):
                
                    self.data = [torch.FloatTensor(row)*3.0/(task['rescale_for_testing']) for row in (data[offset:offset + num_samples])]
                    # print("self.data rescaled", self.data[0])
                
                
                                
        else:
            if(task['variation_type'] =='graph_size'):
                self.data = self.generate_uniform_tsp_data(num_samples, task['graph_size'], task['low'], task['high'])

            if (task['variation_type'] == 'scale'):
                self.data = self.generate_uniform_tsp_data(num_samples, task['graph_size'], task['low'], task['high'])

            if(task['variation_type'] =='distribution'):
                self.data = self.generate_GM_tsp_data_grid(num_samples, task['graph_size'],task['num_modes'])



        self.size = len(self.data)

    def generate_GM_tsp_data_grid(self, dataset_size, tsp_size, num_modes=-1, low=0, high=1):
        # "# GMM-9: each mode with N points; overall clipped to the 0-1 square\n",
        # "# sc: propto stdev of modes arounf the perfect grid; sc1: stdev at each mode\n",
        # print("num modes ", num_modes)

        dataset = []

        remaining_elements = tsp_size

        for i in range(dataset_size):

            # dataset
            cur_gauss = np.empty([0, 2])
            remaining_elements = tsp_size

            modes_done = 0

            sc = 1. / 9.
            sc1 = .045

            elements_in_this_mode = remaining_elements


            rng = default_rng()
            z = array((1., 3., 5.)) / 6
            z = array(meshgrid(z, z))  # perfect grid\n",
            z += rng.uniform(-sc, sc, size=z.shape)  # shake it a bit\n",

            z = z.reshape(2,9)

            cells_chosen = np.random.choice(9, num_modes, replace=False)

            mu_x_array = []
            mu_y_array = []
            for mode in cells_chosen:
                # grid_x = mode//3
                # grid_y = mode % 3
                mu_x = z[0][mode]
                mu_y = z[1][mode]
                mu_x_array.append(mu_x)
                mu_y_array.append(mu_y)

                elements_in_this_mode = int(remaining_elements / (num_modes - modes_done))

                samples_x = scipy.stats.truncnorm.rvs(
                            (low - mu_x) / sc1, (high - mu_x) / sc1, loc=mu_x, scale=sc1, size=elements_in_this_mode)

                samples_y = scipy.stats.truncnorm.rvs(
                    (low - mu_y) / sc1, (high - mu_y) / sc1, loc=mu_y, scale=sc1, size=elements_in_this_mode)
        #

                samples = np.stack((samples_x, samples_y), axis=1)

                cur_gauss = np.concatenate((cur_gauss, samples))

                # elements_in_this_mode = int(elements_in_this_mode)
                remaining_elements = remaining_elements - elements_in_this_mode
                modes_done += 1
                # print(cur_gauss)

            data = torch.Tensor(cur_gauss)
            data = data.reshape(tsp_size, 2)
            dataset.append(data)

        return dataset


    def generate_uniform_tsp_data(self, dataset_size, tsp_size, low, high):
        return  [torch.FloatTensor(tsp_size, 2).uniform_(low, high) for i in range(dataset_size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
