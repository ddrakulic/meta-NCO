import pickle5 as pickle
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy_file", type=str, help="directory with data")
    parser.add_argument("--lkh_file", type=str, help="directory with data")
    parser.add_argument("--capacity", type=int, help="directory with data")
    args = parser.parse_args()

    greedy_file = args.greedy_file
    lkh_file = args.lkh_file
    capacity = args.capacity

    with open(greedy_file, "rb") as file:
        greedy = pickle.load(file)
    with open(lkh_file, "rb") as file:
        lkh = pickle.load(file)

    assert len(greedy) == len(lkh[0])

    tour_len_lkh, tour_len_greedy = 0, 0
    time = 0
    for instance in range(len(greedy[0])):
        tour_len_greedy += greedy[0][instance][0]
        tour_len_lkh += lkh[0][instance][0]
        time += greedy[0][instance][2]

    tour_len_greedy = tour_len_greedy / len(greedy[0])
    tour_len_lkh = tour_len_lkh / len(lkh[0])
    print(tour_len_lkh, tour_len_greedy, time // 60)
