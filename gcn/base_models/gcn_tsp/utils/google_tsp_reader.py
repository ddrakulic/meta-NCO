import os
from random import sample
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
import sklearn
import pickle


class GoogleTSPReader(object):
    """Iterator that reads TSP dataset files and yields mini-batches.
    
    Format expected as in Vinyals et al., 2015: https://arxiv.org/abs/1506.03134, http://goo.gl/NDcOIG
    """

    def __init__(self, num_nodes, num_neighbors, batch_size, filepath, dataset_size=-1, split_details=None,
                 num_batches=-1, shuffle=True):
        """
        Args:
            num_nodes: Number of nodes in TSP tours
            num_neighbors: Number of neighbors to consider for each node in graph
            batch_size: Batch size
            filepath: Path to dataset file (.txt file)
        """
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.filepath = filepath
        self.selected_lines = list()

        offsets = self.read_or_create_offsets()

        if num_batches == 0:
            self.max_iter = 00
            self.filedata = list()
            return
        file = open(filepath, "r")
        if split_details is None or num_batches * batch_size == dataset_size:
            #load whole database
            self.filedata = file.readlines()
            if shuffle:
                self.filedata = sklearn.utils.shuffle(self.filedata)
            if num_batches != -1:
                self.filedata = self.filedata[:num_batches * batch_size]

        elif num_batches == -1:
            # read all data
            split_position = int(dataset_size * split_details["percentage"])
            self.filedata = list()
            if split_details["part"] == 0:
                for _ in range(split_position):
                    line = file.readline()
                    self.filedata.append(line)
            else:
                file.seek(offsets[split_position])
                for _ in range(dataset_size - split_position):
                    line = file.readline()
                    self.filedata.append(line)

            if shuffle:
                self.filedata = sklearn.utils.shuffle(self.filedata)
        else:
            assert (num_batches * batch_size <= dataset_size)
            # read just expected number of batches
            num_samples = int(batch_size * num_batches)
            if split_details["part"] == 0:
                start_idx = 0
                total_num_samples = int(dataset_size * split_details["percentage"])
            else:
                start_idx = int(dataset_size * split_details["percentage"])
                total_num_samples = dataset_size - int(dataset_size * split_details["percentage"])

            if shuffle:
                random_lines = sample(range(0, total_num_samples), num_samples)
            else:
                random_lines = range(0, num_samples)

            random_lines = [el + start_idx for el in random_lines]
            self.filedata = list()
            for line_num in random_lines:
                file.seek(offsets[line_num])
                line = file.readline()
                self.filedata.append(line)
        self.max_iter = (len(self.filedata) // batch_size)

    def __iter__(self):
        for batch in range(self.max_iter):
            start_idx = batch * self.batch_size
            end_idx = (batch + 1) * self.batch_size
            yield self.process_batch(self.filedata[start_idx:end_idx])

    def read_or_create_offsets(self):
        offset_file = self.filepath + ".offsets"
        if not os.path.exists(offset_file):
            # there is no pickle file, create it
            file = open(self.filepath, "r")
            lines = file.readlines()
            offsets = [0]
            offset = 0
            for line in lines:
                offset += len(line)
                offsets.append(offset)
            with open(offset_file, "wb") as file:
                pickle.dump(offsets, file)
        else:
            with open(offset_file, "rb") as file:
                offsets = pickle.load(file)
        return offsets

    def process_batch(self, lines):
        """Helper function to convert raw lines into a mini-batch as a DotDict.
        """
        batch_edges = []
        batch_edges_values = []
        batch_edges_target = []  # Binary classification targets (0/1)
        batch_nodes = []
        batch_nodes_target = []  # Multi-class classification targets (`num_nodes` classes)
        batch_nodes_coord = []
        batch_tour_nodes = []
        batch_tour_len = []

        for line_num, line in enumerate(lines):
            line = line.split(" ")  # Split into list

            # Compute signal on nodes
            nodes = np.ones(self.num_nodes)  # All 1s for TSP...

            # Convert node coordinates to required format
            nodes_coord = []
            for idx in range(0, 2 * self.num_nodes, 2):
                try:
                    nodes_coord.append([float(line[idx]), float(line[idx + 1])])
                except:
                    print(line[idx], line[idx+1])
            # Compute distance matrix

            W_val = squareform(pdist(nodes_coord, metric='euclidean'))
            
            # Compute adjacency matrix
            if self.num_neighbors == -1:
                W = np.ones((self.num_nodes, self.num_nodes))  # Graph is fully connected
            else:
                W = np.zeros((self.num_nodes, self.num_nodes))
                # Determine k-nearest neighbors for each node
                knns = np.argpartition(W_val, kth=self.num_neighbors, axis=-1)[:, self.num_neighbors::-1]
                # Make connections 
                for idx in range(self.num_nodes):
                    W[idx][knns[idx]] = 1
            np.fill_diagonal(W, 2)  # Special token for self-connections
            
            # Convert tour nodes to required format
            # Don't add final connection for tour/cycle
            tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]][:-1]
            
            # Compute node and edge representation of tour + tour_len
            tour_len = 0
            nodes_target = np.zeros(self.num_nodes)
            edges_target = np.zeros((self.num_nodes, self.num_nodes))
            for idx in range(len(tour_nodes) - 1):
                i = tour_nodes[idx]
                j = tour_nodes[idx + 1]
                nodes_target[i] = idx  # node targets: ordering of nodes in tour
                edges_target[i][j] = 1
                edges_target[j][i] = 1
                tour_len += W_val[i][j]
            
            # Add final connection of tour in edge target
            nodes_target[j] = len(tour_nodes) - 1
            edges_target[j][tour_nodes[0]] = 1
            edges_target[tour_nodes[0]][j] = 1
            tour_len += W_val[j][tour_nodes[0]]
            
            # Concatenate the data
            batch_edges.append(W)
            batch_edges_values.append(W_val)
            batch_edges_target.append(edges_target)
            batch_nodes.append(nodes)
            batch_nodes_target.append(nodes_target)
            batch_nodes_coord.append(nodes_coord)
            batch_tour_nodes.append(tour_nodes)
            batch_tour_len.append(tour_len)

        # From list to tensors as a DotDict
        batch = dict()
        batch["edges"] = torch.LongTensor(np.stack(batch_edges, axis=0))
        batch["edges_values"] = torch.FloatTensor(np.stack(batch_edges_values, axis=0))
        batch["edges_target"] = torch.LongTensor(np.stack(batch_edges_target, axis=0))
        batch["nodes"] = torch.LongTensor(np.stack(batch_nodes, axis=0))
        batch["nodes_target"] = torch.LongTensor(np.stack(batch_nodes_target, axis=0))
        batch["nodes_coord"] = torch.FloatTensor(np.stack(batch_nodes_coord, axis=0))
        batch["tour_nodes"] = torch.LongTensor(np.stack(batch_tour_nodes, axis=0))
        batch["tour_len"] = torch.FloatTensor(np.stack(batch_tour_len, axis=0))

        if torch.cuda.is_available():
            batch["edges"] = batch["edges"].cuda()
            batch["edges_values"] = batch["edges_values"].cuda()
            batch["edges_target"] = batch["edges_target"].cuda()
            batch["nodes"] = batch["nodes"].cuda()
            batch["nodes_target"] = batch["nodes_target"].cuda()
            batch["nodes_coord"] = batch["nodes_coord"].cuda()
            batch["tour_nodes"] = batch["tour_nodes"].cuda()
            batch["tour_len"] = batch["tour_len"].cuda()

        return batch


if __name__ == "__main__":
    dataset = GoogleTSPReader(20, -1, 20, "../../data/gcn/tsp20_train_concorde.txt",
                              split_details={"percentage": 0.5, "part": 1}, dataset_size=1000000,
                              num_batches=1)

    print(dataset.max_iter)

