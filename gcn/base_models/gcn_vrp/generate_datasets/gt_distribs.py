import pickle
import argparse
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../../data/gcn_vrp/capacity/")
    parser.add_argument("--input_file", type=str, default="vrp50_capacity40_test.pkl")
    args = parser.parse_args()

    with open(os.path.join(args.input_dir, args.input_file), "rb") as file:
        data = pickle.load(file)
    data = data[:10000]
    tour_lengths, num_tours, tour_capacity = list(), dict(), dict()

    for el in data:
        sol = el[1]
        tour_lengths.append(sol[0])
        nb_tours = sol[1].count(0) + 1
        if nb_tours not in num_tours:
            num_tours[nb_tours] = 1
        else:
            num_tours[nb_tours] += 1

        tour_cap = 0
        for node in el[1][1]:
            if node == 0:
                if tour_cap not in tour_capacity:
                    tour_capacity[tour_cap] = 1
                else:
                    tour_capacity[tour_cap] += 1
                tour_cap = 0
                continue
            tour_cap += el[0][2][node - 1]

        if tour_cap not in tour_capacity:
            tour_capacity[tour_cap] = 1
        else:
            tour_capacity[tour_cap] += 1

    plt.hist(tour_lengths, bins=30, weights=np.ones(len(tour_lengths)) / len(tour_lengths))
    plt.title("Tour length, " + args.input_file + ".png")
    plt.savefig(args.input_file + "_tour_lengths.png")

    plt.clf()
    num_tours = OrderedDict(sorted(num_tours.items()))
    plt.barh(list(num_tours.keys()), list(num_tours.values()))
    for k, v in num_tours.items():
        plt.text(v + 10, k - .1, str(v))
    plt.title("Nb tours, " + args.input_file + ".png")
    plt.savefig(args.input_file + "_nb_tours.png")

    plt.clf()
    tour_capacity = OrderedDict(sorted(tour_capacity.items()))
    plt.barh(list(tour_capacity.keys()), list(tour_capacity.values()))
    plt.title("Total demand per tour, " + args.input_file + ".png")
    plt.savefig(args.input_file + "_demand_per_tour.png")
