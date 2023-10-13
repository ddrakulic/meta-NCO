import os
import glob
import pickle

import numpy as np

DIR = "../data/cvrplib/"


for directory in os.listdir(DIR):
    for path_and_file in glob.glob(DIR + directory + "/*.vrp"):
        filename = path_and_file.split("/")[-1]
        path = path_and_file[:len(path_and_file) - len(filename)]
        print(path_and_file, filename)

        with open(path_and_file, "r") as file:
            lines = file.readlines()
        dimension, capacity, depot_idx = -1, -1, -1
        in_coordinates, in_demands, in_depot = False, False, False
        original_coordinates, demands = list(), list()
        for line in lines:
            line_list = line.split()
            if line_list[0][:9] == "DIMENSION":
                dimension = int(line.split()[-1])
            if line_list[0][:8] == "CAPACITY":
               capacity = int(line.split()[-1])

            if line_list[0] == "NODE_COORD_SECTION":
                in_coordinates = True
                continue

            if line_list[0] == "DEMAND_SECTION":
                in_coordinates = False
                in_demands = True
                continue

            if line_list[0] == "DEPOT_SECTION":
                in_demands = False
                in_depot = True
                continue

            if in_coordinates:
                original_coordinates.append([float(line_list[-2]), float(line_list[-1])])

            if in_demands:
                demands.append(int(line_list[-1]))

            if in_depot:
                depot_idx = int(line_list[0])
                break

        assert dimension > 0
        assert capacity > 0
        assert depot_idx > 0
        assert len(original_coordinates) == dimension
        assert len(demands) == dimension
        assert demands[depot_idx - 1] == 0
        original_coordinates = np.array(original_coordinates)

        normalized_coordinates = (original_coordinates - original_coordinates.min()) / \
                                 (original_coordinates.max() - original_coordinates.min())

        scale = min(1 - (normalized_coordinates[:, 0].max() - normalized_coordinates[:, 0].min()),
                    1 - (normalized_coordinates[:, 1].max() - normalized_coordinates[:, 1].min()))

        if scale == 1 - (normalized_coordinates[:, 0].max() - normalized_coordinates[:, 0].min()):
            if scale == 0:
                normalized_coordinates[:, 1] -= normalized_coordinates[:, 1].min()
            # scaling over x
            if normalized_coordinates[:, 1].min() == 0:
                normalized_coordinates[:, 1] += scale
        else:
            if scale == 0:
                normalized_coordinates[:, 0] -= normalized_coordinates[:, 0].min()
            if normalized_coordinates[:, 0].min() == 0:
                normalized_coordinates[:, 0] += scale

        normalized_coordinates = (normalized_coordinates - scale) / (1 - scale)

        normalized_depot = normalized_coordinates[depot_idx - 1]
        normalized_coordinates = np.delete(normalized_coordinates, depot_idx-1, axis=0)
        del demands[depot_idx-1]
        normalized_result, original_result = list(), list()
        normalized_result.append(normalized_depot.tolist())
        normalized_result.append(normalized_coordinates.tolist())
        normalized_result.append(demands)
        normalized_result.append(float(capacity))
        normalized_result = tuple(normalized_result)

        original_depot = original_coordinates[depot_idx - 1]
        original_coordinates = np.delete(original_coordinates, depot_idx-1, axis=0)
        original_result.append(original_depot.tolist())
        original_result.append(original_coordinates.tolist())
        original_result.append(demands)
        original_result.append(float(capacity))
        original_result = tuple(original_result)

        with open(path + "pickled/" + filename + "_original.pkl", "wb") as file:
            pickle.dump(original_result, file)
        with open(path + "pickled/" + filename + "_normalized.pkl", "wb") as file:
            pickle.dump(normalized_result, file)

        print(path_and_file, "done.")

