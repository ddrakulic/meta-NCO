import os
import time
import argparse
import pprint as pp
from os import unlink

import numpy as np
from concorde.tsp import TSPSolver
import multiprocessing as mp


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


def solve_and_save(args):
    nodes_coords, filename = args
    with open(filename, "w") as f:
        for nodes_coord in nodes_coords:
            solver = TSPSolver.from_data(nodes_coord[:, 0]*1e6, nodes_coord[:, 1]*1e6, norm="EUC_2D")
            solution = solver.solve()
            f.write(" ".join( str(x)+str(" ")+str(y) for x, y in nodes_coord))
            f.write(str(" ") + str('output') + str(" "))
            f.write(str(" ").join( str(node_idx+1) for node_idx in solution.tour))
            f.write(str(" ") + str(solution.tour[0]+1) + str(" "))
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_threads", type=int, default=-1)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--num_nodes", type=int, default=50)
    parser.add_argument("--num_modes", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="../../../data/gcn_tsp/")
    parser.add_argument("--seed", type=int, default=0)
    opts = parser.parse_args()

    if opts.filename is None:
        filename = f"tsp_{opts.num_nodes}_{opts.num_modes}_{opts.scale:g}_seed{opts.seed}_concorde.txt"
    else:
        filename = opts.filename

    np.random.seed(opts.seed)

    filename = os.path.join(opts.output_dir, filename)

    if opts.num_threads == -1:
        num_threads = mp.cpu_count()
    else:
        num_threads = min(opts.num_threads, mp.cpu_count())
    print("Generating in ", num_threads, "threads")
    tmp_filenames = [filename + '.thread' + str(x) for x in range(num_threads)]

    # Pretty print the run args
    pp.pprint(vars(opts))

    if opts.num_modes == 0:
        all_nodes_coord = opts.scale * np.random.random([opts.num_samples, opts.num_nodes, 2])
    else:
        all_nodes_coord, _ = grid(opts.num_nodes, opts.num_modes, opts.scale, opts.num_samples)

    chunk_size = opts.num_samples // num_threads
    chunks = list()
    for chunk in range(num_threads - 1):
        chunks.append(all_nodes_coord[chunk * chunk_size: (chunk + 1) * chunk_size])
    chunks.append(all_nodes_coord[chunk_size * (num_threads - 1):])

    start_time = time.time()
    pool = mp.Pool(num_threads)
    pool.map(solve_and_save, list(zip(chunks, tmp_filenames)))

    pool.close()
    pool.join()

    # concatenate all tmp files and delete them
    with open(filename, 'w') as outfile:
        for input_filename in tmp_filenames:
            with open(input_filename) as infile:
                for line in infile:
                    outfile.write(line)
                unlink(input_filename)
    end_time = time.time() - start_time

    print(f"Completed generation of {opts.num_samples} samples of TSP{opts.num_nodes}.")
    print(f"Total time: {end_time:.1f}s")
    print(f"Average time: {(end_time)/opts.num_samples:.1f}s")
