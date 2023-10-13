import os.path
import numpy as np
import argparse
import pickle
import time
import multiprocessing as mp
from vrp_baselines import get_lkh_executable, solve_lkh_log


def grid(num_nodes: int, num_modes: int, scale: float = None, num_samples: int = None,
         base=(lambda x: np.array(np.meshgrid(x, x)).T.reshape((9, 2)))(np.array((1., 3., 5.))/6.),
         sc_=1./9.,
         sc=.6/9.):
    assert 0 < num_modes <= len(base)

    def g(p=np.ones(num_modes)/num_modes):
        z_ = base[np.random.choice(9, num_modes, False)]+sc_*np.random.uniform(-1, 1, size=(num_modes, 2)) # subsample & shake the base
        z = np.repeat(z_, np.random.multinomial(num_nodes, p), axis=0) # sample at each mode
        return z, z_
    z, z_ = g() if num_samples is None else map(np.stack, zip(*(g() for _ in range(num_samples))))
    z += sc*np.random.normal(size=z.shape)
    np.clip(z, a_min=0, a_max=1, out=z) # clip to 0-1 square, but maybe we don't need that
    if scale is not None:
        z *= scale
    return z, z_


def generate_vrp_data(dataset_size, num_nodes, num_modes, capacity):
    if num_modes == 0:
        return list(zip(
            np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
            np.random.uniform(size=(dataset_size, num_nodes, 2)).tolist(),  # Node locations
            np.random.randint(1, 10, size=(dataset_size, num_nodes)).tolist(),  # Demand, uniform integer 1 ... 9
            np.full(dataset_size, capacity).tolist()  # Capacity, same for whole dataset
        ))
    else:
        all_nodes_coord, _ = grid(num_nodes, num_modes, 1, dataset_size)
        return list(zip(
            np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
            all_nodes_coord.tolist(), # Node locations
            np.random.randint(1, 10, size=(dataset_size, num_nodes)).tolist(),  # Demand, uniform integer 1 ... 9
            np.full(dataset_size, capacity).tolist()  # Capacity, same for whole dataset
        ))


def prepare_and_solve_lkh_log(executable, output_dir, name, ds):
    solution = solve_lkh_log(executable, output_dir, name, ds[0], ds[1], ds[2], ds[3])
    return ds, solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_threads", type=int, default=-1)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--num_nodes", type=int, default=30)
    parser.add_argument("--num_modes", type=int, default=0)
    parser.add_argument("--capacity", type=float, default=20.0)
    parser.add_argument("--filename", type=str, default="vrp100_train.pkl")
    parser.add_argument("--output_dir", type=str, default="/home/ddrakuli/Projects/meta-nco/data/gcn_vrp/test")
    args = parser.parse_args()

    n_proc = mp.cpu_count()
    if args.num_threads != -1:
        n_proc = min(n_proc, args.num_threads)

    if os.path.exists(args.output_dir):
        print("Path already exists, dataset will be added.")
        if not os.path.exists(os.path.join(args.output_dir, "lkh")):
            exit("Something is wrong, lkh dir does not exist.")
    else:
        os.mkdir(args.output_dir)
        os.mkdir(os.path.join(args.output_dir, "lkh"))

    dataset = generate_vrp_data(args.num_samples, args.num_nodes, args.num_modes, args.capacity)

    executable = get_lkh_executable()
    start_time = time.time()
    with mp.Pool(processes=n_proc) as pool:
        proc_results = [pool.apply_async(prepare_and_solve_lkh_log, args=(executable,
                                                                          os.path.join(args.output_dir, "lkh"),
                                                                          str(i).zfill(4), ds))
                        for i, ds in enumerate(dataset)]
        results = [result.get() for result in proc_results]
    total_time = time.time() - start_time
    results_with_solution = list()
    results_without_solution = list()
    total_solving_time = 0
    for el in results:
        if el[1] is not None:
            results_with_solution.append(el)
            total_solving_time += el[1][2]
        else:
            results_without_solution.append((el[0]))

    with open(os.path.join(args.output_dir, args.filename), "wb") as file:
        pickle.dump(results_with_solution, file)
    with open(os.path.join(args.output_dir, args.filename + "no_sol"), "wb") as file:
        pickle.dump(results_without_solution, file)

    # print statistics
    print("Nb nodes:", args.num_nodes, "Capacity:", args.capacity,
          "Nb samples:", len(results_with_solution), "/", args.num_samples)
    print("Total computation time", total_solving_time)
    print("Running computation time per instance", total_time / len(results_with_solution))
    print("Total running time", total_time)
