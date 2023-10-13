import pickle5 as pickle
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="directory with data")
    args = parser.parse_args()

    prefix = args.data_dir.split("/")[-2]
    for suffix in ["train", "val", "test"]:
        instances, solutions = list(), list()
        with open(os.path.join(args.data_dir, prefix + "_" + suffix + ".pkl"), "rb") as file:
            data = pickle.load(file)
        for row in data:
            instances.append(row[0])
            solutions.append(row[1])
        solutions = (solutions, 1)

        with open(os.path.join(args.data_dir, prefix + "_" + suffix + "-instances.pkl"), "wb") as file:
            pickle.dump(instances, file)
        with open(os.path.join(args.data_dir, prefix + "_" + suffix + "-solutions.pkl"), "wb") as file:
            pickle.dump(solutions, file)

    print("Done.")

