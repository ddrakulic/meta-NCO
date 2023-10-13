import pickle5 as pickle
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from scipy.special import softmax

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=int, default=3)
    parser.add_argument("--dir", type=str, default="data/gcn_vrp/vrp_nazari100_test_seed1234/")
    parser.add_argument("--filename_problem", type=str, default="vrp_nazari100_test_seed1234.pkl")
    parser.add_argument("--filename_heatmap", type=str, default="heatmaps_vrp_nazari100.pkl")

    args = parser.parse_args()

    with open(os.path.join(args.dir, args.filename_problem), "rb") as file:
        data_problem = pickle.load(file)
    with open(os.path.join(args.dir, args.filename_heatmap), "rb") as file:
        data_heatmap = pickle.load(file)

    for i in range(200):
        problem = data_problem[i]
        heatmap = softmax(data_heatmap[0][i])

        plt.clf()
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        depot = [problem[0][0], problem[0][1]]
        coords = np.array(problem[1])

        plt.plot(depot[0], depot[1], "ro")
        plt.plot(coords[:, 0], coords[:, 1], "o")
        coords = np.insert(coords, 0, depot, axis=0)

        for idx in range(len(coords)):
            idy1 = np.argmax(heatmap[idx])
            heatmap[idx][idy1] = 1e-100
            idy2 = np.argmax(heatmap[idx])
            heatmap[idx][idy2] = 1e-100
            idy3 = np.argmax(heatmap[idx])

            plt.plot((coords[idx][0], coords[idy1][0]), (coords[idx][1], coords[idy1][1]), color="0")
            plt.plot((coords[idx][0], coords[idy2][0]), (coords[idx][1], coords[idy2][1]), color="0.5")

        plt.savefig("gcn_vrp/plots/dpdp/heatmap_" + str(i) + ".png")
    print("Done.")
