import numpy as np
import pickle


DEFAULT_INT = np.int64
DEFAULT_FLOAT = np.float64


def load(filename):
    pickled_obj_file = open(filename, 'rb')
    obj = pickle.load(pickled_obj_file)
    pickled_obj_file.close()

    return obj


def save(obj, filename):
    """
    objectives and their args are not saved with the ES
    """
    pickled_obj_file = open(filename, 'wb')
    pickle.dump(obj, pickled_obj_file, protocol=pickle.DEFAULT_PROTOCOL)
    pickled_obj_file.close()


def normalize_root_mean_squared_error(residuals, ymax, ymin):
    """
    :param residuals: a numpy array of the difference between target and
        model. THIS ARRAY IS MODIFIED IN PLACE
    :param ymax: largest observed value
    :param ymin: smallest observed value
    :return: NRMSE
    """
    np.power(residuals, 2, out=residuals)
    return np.sqrt(np.mean(residuals)) / (ymax - ymin)


def absolute_error(residuals):
    """
    :param residuals: a numpy array of the difference between target and
        model. THIS ARRAY IS MODIFIED IN PLACE
    :return: AE
    """

    np.absolute(residuals, out=residuals)
    return np.sum(residuals)


def run_mpi_experiment(f, parameters, savefile=None, return_results=False):
    """
    Only use this if mpi4py is installed. This function is for allowing users
    to run experiments over many cpu's with a range of parameter settings (say
    to produce plots or contours of how results change with one or more
    parameters). f should be a function or partial function that just takes
    a set of parameters as an argument and returns some result. Parameters should
    be a list of parameter sets, which will be distributed over the nodes.

    The run involves executing each parameter set and gathering the results
    from the various nodes into a list, where each result matches its
    corresponding parameter set in the given list. If the user wants to "loop"
    over multiple parameters, then flatten that list and provide the flattened
    list as an argument, then unflatten the result list into the desired shape.

    The result list can be pickled and saved by the root node into a file
    if a filename is provided. Optionally the results can be returned from
    the function.

    :param f: a function that results some result and takes a single argument,
        a parameter set/dict/etc.
    :param parameters: a list of parameters
    :param savefile: name of file to save results. Default: None (doesn't save
        if set to None)
    :param return_results: Default: False
    :return: None, optionally a list of results
    """

    import math
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # This splits the parameter list into sublists for each rank
    chunksize = int(math.ceil(len(parameters) / size))
    tasks_by_rank = [parameters[i*chunksize: i*chunksize+chunksize]
                     for i in range(size)]

    # Even if results are objects of varying length or uneven, we can glob them
    # up and slap them in the result list.
    globbed_results = []
    for parameters in tasks_by_rank[rank]:
        globbed_results.append(f(parameters))

    all_results = [[] for i in range(size)]
    all_results = comm.gather(globbed_results, root=0)

    # Save and/or return results
    if rank == 0:
        # Unglob results so they are same length as parameters and in correct order
        all_results = [result for glob in all_results for result in glob]

        if not (savefile is None):
            save(all_results, savefile)

        if return_results:
            return all_results
