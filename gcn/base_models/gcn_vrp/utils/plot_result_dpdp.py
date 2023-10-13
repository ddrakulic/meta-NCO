import pickle5 as pickle
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import matplotlib.colors as mcolors

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="data/gcn_vrp/vrp_nazari100_test_seed1234/")
    parser.add_argument("--filename_problem", type=str, default="vrp_nazari100_test_seed1234.pkl")
    parser.add_argument("--filename_solution", type=str, default="vrp_nazari100_test_seed1234-greedy.pkl")
    args = parser.parse_args()

    with open(os.path.join(args.dir, args.filename_problem), "rb") as file:
        data_problem = pickle.load(file)
    with open(os.path.join(args.dir, args.filename_solution), "rb") as file:
        data_solution = pickle.load(file)

    for i in range(200):
        problem = data_problem[i]
        solution = data_solution[0][i]

        plt.clf()
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        depot = [problem[0][0], problem[0][1]]
        coords = np.array(problem[1])

        plt.plot(depot[0], depot[1], "ro")
        plt.plot(coords[:, 0], coords[:, 1], "o")
        for idx, location in enumerate(coords):
            plt.text(location[0] + 0.01, location[1], problem[2][idx])
        coords = np.insert(coords, 0, depot, axis=0)

        tours_sol = [0] + solution[1] + [0]
        tour_num = 0
        for idx in range(len(tours_sol) - 1):
            if tours_sol[idx] == 0:
                color = list(mcolors.TABLEAU_COLORS.values())[tour_num % len(mcolors.TABLEAU_COLORS.values())]
                tour_num += 1
            plt.plot((coords[tours_sol[idx]][0], coords[tours_sol[idx + 1]][0]),
                     (coords[tours_sol[idx]][1], coords[tours_sol[idx + 1]][1]),
                     color=color)

        plt.savefig("gcn_vrp/plots/dpdp/solution_" + str(i) + "_greedy.png")
    print("Done.")
