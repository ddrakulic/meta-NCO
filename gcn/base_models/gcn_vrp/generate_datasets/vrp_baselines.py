import os
import math
import numpy as np
from data_utils import load_dataset, save_dataset
from subprocess import check_call, check_output
from urllib.parse import urlparse
import tempfile
import time


def get_lkh_executable(url="http://www.akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.4.tgz"):

    cwd = os.path.abspath(os.path.join("problems", "vrp", "lkh"))
    os.makedirs(cwd, exist_ok=True)

    file = os.path.join(cwd, os.path.split(urlparse(url).path)[-1])
    filedir = os.path.splitext(file)[0]

    if not os.path.isdir(filedir):
        print("{} not found, downloading and compiling".format(filedir))

        check_call(["wget", url], cwd=cwd)
        assert os.path.isfile(file), "Download failed, {} does not exist".format(file)
        check_call(["tar", "xvfz", file], cwd=cwd)

        assert os.path.isdir(filedir), "Extracting failed, dir {} does not exist".format(filedir)
        check_call("make", cwd=filedir)
        os.remove(file)

    executable = os.path.join(filedir, "LKH")
    assert os.path.isfile(executable)
    return os.path.abspath(executable)


def solve_lkh(executable, depot, loc, demand, capacity):
    with tempfile.TemporaryDirectory() as tempdir:
        problem_filename = os.path.join(tempdir, "problem.vrp")
        output_filename = os.path.join(tempdir, "output.tour")
        param_filename = os.path.join(tempdir, "params.par")

        starttime = time.time()
        write_vrplib(problem_filename, depot, loc, demand, capacity)
        params = {"PROBLEM_FILE": problem_filename, "OUTPUT_TOUR_FILE": output_filename}
        write_lkh_par(param_filename, params)
        output = check_output([executable, param_filename])
        result = read_vrplib(output_filename, n=len(demand))
        duration = time.time() - starttime
        return result, output, duration


def solve_lkh_log(executable, directory, name, depot, loc, demand, capacity,
                  grid_size=1, mask=None, runs=1, unlimited_routes=False, disable_cache=False, only_cache=False):

    # lkhu = LKH with unlimited routes/salesmen
    alg_name = 'lkhu' if unlimited_routes else 'lkh'
    basename = "{}.{}{}".format(name, alg_name, runs)
    problem_filename = os.path.join(directory, "{}.vrp".format(basename))
    tour_filename = os.path.join(directory, "{}.tour".format(basename))
    output_filename = os.path.join(directory, "{}.pkl".format(basename))
    param_filename = os.path.join(directory, "{}.par".format(basename))
    log_filename = os.path.join(directory, "{}.log".format(basename))
    if mask is not None:
        edges_filename = os.path.join(directory, "{}.edges".format(basename))

    try:
        # May have already been run
        if os.path.isfile(output_filename) and not disable_cache:
            tour, duration = load_dataset(output_filename)
        elif not only_cache:
            write_vrplib(problem_filename, depot, loc, demand, capacity, grid_size, name=name)

            params = {
                "PROBLEM_FILE": problem_filename,
                "OUTPUT_TOUR_FILE": tour_filename,
                "RUNS": runs,
                "SEED": 1234
            }
            if unlimited_routes:
                # Option unlimited_routes is False by default for backwards compatibility
                # By default lkh computes some bound for the number of salesmen which can result in infeasible solutions
                params['SALESMEN'] = len(loc)
                params['MTSP_MIN_SIZE'] = 0  # We should allow for empty routes
            if mask is not None:
                # assert unlimited_routes, "For now, edges only work with unlimited routes so we now num routes a priori"
                if unlimited_routes:
                    num_depots = len(loc)
                else:
                    # Standard LKH computes a bound
                    num_depots = math.ceil(sum(demand) / capacity)
                write_vrp_edges(edges_filename, mask, num_depots=num_depots)
                params['EDGE_FILE'] = edges_filename
                # Next to the predicted edges, we should not have additional edges (nearest neighbours etc.)
                params['MAX_CANDIDATES'] = 0
            write_lkh_par(param_filename, params)

            with open(log_filename, 'w') as f:
                start = time.time()
                check_call([executable, param_filename], stdout=f, stderr=f)
                duration = time.time() - start

            tour = read_vrplib(tour_filename, n=len(demand))

            save_dataset((tour, duration), output_filename)
        else:
            raise Exception("No cached solution found")

        return calc_vrp_cost(depot, loc, tour), tour, duration

    except Exception as e:
        print("Exception occured")
        print(e)
        return None


def calc_vrp_cost(depot, loc, tour):
    assert (np.sort(tour)[-len(loc):] == np.arange(len(loc)) + 1).all(), "All nodes must be visited once!"
    # TODO validate capacity constraints
    loc_with_depot = np.vstack((np.array(depot)[None, :], np.array(loc)))
    sorted_locs = loc_with_depot[np.concatenate(([0], tour, [0]))]
    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()


def write_lkh_par(filename, parameters):
    default_parameters = {  # Use none to include as flag instead of kv
        "SPECIAL": None,
        "MAX_TRIALS": 10000,
        "RUNS": 10,
        "TRACE_LEVEL": 1,
        "SEED": 0
    }
    with open(filename, 'w') as f:
        for k, v in {**default_parameters, **parameters}.items():
            if v is None:
                f.write("{}\n".format(k))
            else:
                f.write("{} = {}\n".format(k, v))


def read_vrplib(filename, n):
    with open(filename, 'r') as f:
        tour = []
        dimension = 0
        started = False
        for line in f:
            if started:
                loc = int(line)
                if loc == -1:
                    break
                tour.append(loc)
            if line.startswith("DIMENSION"):
                dimension = int(line.split(" ")[-1])

            if line.startswith("TOUR_SECTION"):
                started = True

    assert len(tour) == dimension
    tour = np.array(tour).astype(int) - 1  # Subtract 1 as depot is 1 and should be 0
    tour[tour > n] = 0  # Any nodes above the number of nodes there are is also depot
    assert tour[0] == 0  # Tour should start with depot
    # Now remove duplicates, remove if node is equal to next one (cyclic)
    tour = tour[tour != np.roll(tour, -1)]
    assert tour[-1] != 0  # Tour should not end with depot
    return tour[1:].tolist()


def write_vrplib(filename, depot, loc, demand, capacity, grid_size, name="problem"):

    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "CVRP"),
                ("DIMENSION", len(loc) + 1),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
                ("CAPACITY", capacity)
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, int(x / grid_size * 100000 + 0.5), int(y / grid_size * 100000 + 0.5))  # VRPlib does not take floats
            #"{}\t{}\t{}".format(i + 1, x, y)
            for i, (x, y) in enumerate([depot] + loc)
        ]))
        f.write("\n")
        f.write("DEMAND_SECTION\n")
        f.write("\n".join([
            "{}\t{}".format(i + 1, d)
            for i, d in enumerate([0] + demand)
        ]))
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")


def write_vrp_edges(filename, mask, num_depots=None):
    n = mask.shape[0] - 1
    if num_depots is None:
        num_depots = n

    # Bit weird but this is how LKH arranges multiple depots
    depots = np.arange(num_depots)  # 0 to n_depots - 1
    depots[1:] += n  # Add num nodes, so depots are (0, n + 1, n + 2, ...)

    with open(filename, 'w') as f:
        # Note: mask is already upper triangular
        # First row is connections to depot, we should replicate these for 'all depots'
        depot_edges = np.flatnonzero(mask[0])
        # TODO remove since this is temporary
        depot_edges = np.arange(n) + 1 # 1 to n, connect depot to every node to ensure feasibility
        frm, to = mask[1:, 1:].nonzero()
        # Add one for stripping of depot, nodes are 1 ... n
        frm, to = frm + 1, to + 1

        num_nodes = n + num_depots
        num_edges = num_depots * len(depot_edges) + len(frm)

        f.write(f"{num_nodes} {num_edges}\n")

        f.write("\n".join([
                              f"{depot} {node}"
                              for depot in depots
                              for node in depot_edges
                          ] + [
                              f"{f} {t}"
                              for f, t in zip(frm, to)
                          ]))

        f.write("\n")
        f.write("EOF\n")
