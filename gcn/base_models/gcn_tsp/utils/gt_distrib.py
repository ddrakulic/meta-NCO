from google_tsp_reader import GoogleTSPReader
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, default=10)
    parser.add_argument('--num_neighbors', type=int, default=-1)
    parser.add_argument('--data_dir', type=str, default="/home/ddrakuli/Projects/meta-nco/data/gcn_tsp/size/")
    parser.add_argument('--filename', type=str, default="tsp10_val_concorde.txt")

    args = parser.parse_args()

    dataset = GoogleTSPReader(num_nodes=args.num_nodes, num_neighbors=args.num_neighbors, batch_size=20,
                              filepath=os.path.join(args.data_dir, args.filename))
    values = list()
    for batch in dataset:
        values += batch["tour_len"].cpu().detach().tolist()

    plt.hist(values, bins=30, weights=np.ones(len(values)) / len(values))
    plt.grid()
    plt.title(args.filename)
    plt.savefig(args.filename[:-3] + "png")
    print(args.filename, "done.")
