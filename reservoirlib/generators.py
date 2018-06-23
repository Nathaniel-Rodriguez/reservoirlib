import numpy as np


DEFAULT_INT = np.int64
DEFAULT_FLOAT = np.float64


def reservoir_as_edge_list(reservoir, dtype1=None, dtype2=None):
    """
    Converts a reservoir to an edge list Ex2 with (i,j) where i->j.
    :param reservoir: reservoir as adjacency matrix
    :param dtype1: type of returned edge list
    :param dtype2: type of returned weight array
    :return: Ex2 numpy array, E numpy array of weights
    """
    raise NotImplementedError  # TODO implement


def generate_weights_for_edge_list(graph, distribution, dtype=None):
    """
    Generates weights for edges in edge list
    :param graph: Ex2 numpy array
    :param distribution: a distribution that can be called with a shape parameter
        e.g.: random_values = distribution(shape)
    :param dtype: type of weight array
    :return: an E numpy array of type dtype
    """

    if dtype is None:
        dtype = DEFAULT_FLOAT

    return distribution(size=len(graph)).astype(dtype)


def generate_reservoir_input_weights(num_inputs, reservoir_size, input_fraction,
                                     distribution, dtype=None, by_dimension=True):
    """
    N=reservoir_size, K=num_inputs
    Generates a NxK numpy array where only a fraction of the reservoir neurons
    receive inputs.
    :param num_inputs: number of input dimensions
    :param reservoir_size: number of neurons in reservoir
    :param input_fraction: fraction of neurons that receive input for each dimension
    :param distribution: a distribution that can be called with a shape parameter
        and with
        e.g.: random_values = distribution(shape)
    :param dtype: the dtype of the input array
    :param by_dimension: whether input_fraction is per dimension or for all inputs.
        If True, each input gets input_fraction of connections.
        If False, all inputs share input_fraction. Possible that some inputs maybe
        disconnected.
        Default: True
    :return: a NxK
    """

    if by_dimension:
        num_weights = int(reservoir_size * input_fraction * num_inputs)
    else:
        num_weights = int(reservoir_size * input_fraction)

    if dtype is None:
        dtype = DEFAULT_FLOAT

    input_array = np.zeros((reservoir_size, num_inputs), dtype=dtype)
    chosen_indices = distribution.choice(np.product(input_array.shape),
                                         num_weights)
    chosen_indices.sort()
    input_array.flat[chosen_indices] = distribution(size=len(chosen_indices))
    return input_array


def generate_adj_reservoir_from_edge_list(graph, distribution, dtype=None,
                                          N=None):
    """
    Generates a weighted reservoir from an unweighted network. The network
    can be expressed as an edge list
    :param graph: a Ex2 numpy edge list. Indices are assumed to correspond to
        neuron ids. Assumed edge list represents i->j connections given (i,j)
        values. e.g. (graph[x][i], graph[x][j])
    :param distribution: a distribution that can be called with a shape parameter
        e.g.: random_values = distribution(shape)
    :param dtype: numpy type of output reservoir
    :param N: if the graph is disconnected, then latter nodes maybe lost. So N
        would need to be specified
    :return: numpy square matrix of size NxN, where N = # of identified neurons
        rows are predecessors of ith neuron, e.g. ith row is all of the neurons
        connecting TO i. So the sum of values in the ith row is the in-strength
        of i. This orientation is chosen so that the dot product: M * x can be
        done to give the sum of incoming excitations if x is the state vector Nx1.
    """

    if dtype is None:
        dtype = DEFAULT_FLOAT

    if N is None:
        N = graph.max()

    reservoir = np.zeros((N, N), dtype=dtype)
    flat_indices = np.ravel_multi_index(np.flip(graph.transpose(), 0), (N, N))
    reservoir.flat[flat_indices] = distribution(size=len(flat_indices))

    return reservoir


def generate_adj_reservoir_from_adj_matrix(graph, distribution, dtype=None):
    """
    Generates a weighted reservoir from an unweighted network. The network
    can be expressed as an numpy matrix
    :param graph: a matrix with head as first axis, tails as second axis.
        e.g. graph[i,j] gives link from j->i
    :param distribution: a distribution that can be called with a shape parameter
        e.g.: random_values = distribution(shape)
    :param dtype: return type of array
    :return: numpy square matrix of size NxN, where N = # of identified neurons
        rows are predecessors of ith neuron, e.g. ith row is all of the neurons
        connecting TO i. So the sum of values in the ith row is the in-strength
        of i. This orientation is chosen so that the dot product: M * x can be
        done to give the sum of incoming excitations if x is the state vector Nx1.
    """

    if dtype is None:
        dtype = DEFAULT_FLOAT

    reservoir = np.zeros(graph.shape, dtype=dtype)
    nonzero_indices = np.nonzero(graph)
    flat_indices = np.ravel_multi_index(nonzero_indices, reservoir.shape)
    reservoir.flat[flat_indices] = distribution(size=len(flat_indices))

    return reservoir


def generate_adj_reservoir_from_nx_graph(graph, distribution, dtype=None):
    """
    Generates a weighted reservoir from an unweighted network. The network
    can be expressed as an numpy matrix
    :param graph: a matrix with head as first axis, tails as second axis.
        e.g. graph[i,j] gives link from j->i
    :param distribution: a distribution that can be called with a shape parameter
        e.g.: random_values = distribution(shape)
    :param dtype: return type of array
    :return: numpy square matrix of size NxN, where N = # of identified neurons
        rows are predecessors of ith neuron, e.g. ith row is all of the neurons
        connecting TO i. So the sum of values in the ith row is the in-strength
        of i. This orientation is chosen so that the dot product: M * x can be
        done to give the sum of incoming excitations if x is the state vector Nx1.
    """

    raise NotImplementedError  # TODO implement


if __name__ == "__main__":
    pass
