import glob
import os
import pickle
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="directory with data")
    args = parser.parse_args()

    file_name = glob.glob(os.path.join(args.data_dir, "*.pkl"))[0].split("/")[-1].split(".")[0].split("-")[0]
    print(file_name)

    res = list()
    for i in range(10):
        path = os.path.join(args.data_dir, file_name + "-" + str(i) + ".pkl")
        with open(path, "rb") as file:
            res1 = pickle.load(file)
        res.extend(res1)
        print(path, "done.")
    with open(os.path.join(args.data_dir, file_name + "_train.pkl"), "wb") as file:
        pickle.dump(res, file)
    print(file_name, len(res))
