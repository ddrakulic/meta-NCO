import pickle5 as pickle
import numpy as np
import time
from scipy.special import softmax
from scipy.spatial.distance import pdist, squareform
import argparse


def find_next_node(origin, heatmap):
    probs = softmax(heatmap)
    next_node = np.argmax(probs[origin])
    return next_node


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="directory with data")
    parser.add_argument("--heatmaps_file", type=str, help="directory with data")
    parser.add_argument("--output_file", type=str, help="directory with data")
    parser.add_argument("--capacity", type=int, help="directory with data")
    args = parser.parse_args()

    capacity = args.capacity

    input_file = args.input_file
    heatmaps_file = args.input_file
    output_file = args.input_file

    with open(input_file, "rb") as file:
        inputs = pickle.load(file)
    with open(heatmaps_file, "rb") as file:
        heatmaps = pickle.load(file)

    assert len(inputs) == len(heatmaps[0])
    res = []

    for instance in range(len(inputs)):
        start_time = time.time()
        demands = inputs[instance][2]
        heatmap = heatmaps[0][instance]
        num_nodes = len(demands)

        coords = [inputs[instance][0]]
        coords.extend(inputs[instance][1])

        dist_matrix = squareform(pdist(coords, metric='euclidean'))

        visited = np.zeros(num_nodes, dtype=np.int32)
        tours = [0]
        tour_capacities = []
        tour_capacity = 0
        while np.sum(visited) != num_nodes:
            current_node = tours[-1]
            next_node = find_next_node(current_node, heatmap)
            if tour_capacity + demands[next_node - 1] > capacity:
                # capacity is fill, go to the depot
                tours.append(0)
                tour_capacities.append(tour_capacity)
                tour_capacity = 0
                continue
            # visit next node
            tour_capacity += demands[next_node - 1]
            visited[next_node - 1] = 1
            heatmap[:, next_node] = -1e3
            heatmap[next_node, current_node] = -1e3

            tours.append(next_node)
        tour_capacities.append(tour_capacity)

        tour_len = 0
        for i in range(1, len(tours)):
            tour_len += dist_matrix[tours[i-1], tours[i]]

        res.append([tour_len, tours, time.time() - start_time])
    # add one element in dim=1, to be in the same shape as result of LKH [solutions, meta info]
    res = [res] + [0]
    with open(output_file, "wb") as file:
        pickle.dump(res, file)
