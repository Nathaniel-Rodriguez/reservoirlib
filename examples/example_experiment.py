import numpy as np
from graphgen.lfr_generators import unweighted_directed_lfr_as_adj
from reservoirlib.distribution import Distribution
from reservoirlib.experiment import BenchmarkExperiment
from reservoirlib.trainer import LeastSquaredErrorTrainer
from reservoirlib.task import BinaryMemoryCapacityTask
from reservoirlib.generator import generate_reservoir_input_weights
from reservoirlib.generator import generate_adj_reservoir_from_adj_matrix
from reservoirlib.esn import DiscreteEchoStateNetwork


def test_experiment():
    """
    This example builds a simple experiment. It requires the graphgen library
    for creating a reservoir, and should be installed. But it is not a dependency
    for reservoirlib. You can build your graphs how you like (using networkx
    for instance).
    """

    # First, choose a desired task and set its parameters
    mc_task = BinaryMemoryCapacityTask(duration=12, cut=2, num_lags=2, shift=2)

    # Now lets generate a base reservoir using graphgen to define the network
    # and reservoirlib's reservoir generators to assign random weights
    # according to a desired distribution which I define below
    dist = Distribution("uniform", {'low': -1.0, 'high': 1.0}, seed=52423)

    # Now for the network parameters
    net_pars = {'num_nodes': 10,
                'average_k': 4,
                'max_degree': 4,
                'mu': 0.5,
                'com_size_min': 10,
                'com_size_max': 10,
                'seed': 2342643,
                'transpose': True,
                'dtype': np.float32}
    graph, members = unweighted_directed_lfr_as_adj(**net_pars)
    reservoir = generate_adj_reservoir_from_adj_matrix(graph, dist)

    # To generate the input weights for our reservoir we can use the task's
    # input_dimensions property, so we know how many inputs there are
    input_weights = generate_reservoir_input_weights(mc_task.input_dimensions,
                                                     reservoir_size=10,
                                                     input_fraction=0.4,
                                                     distribution=dist,
                                                     by_dimension=True)

    # We will be using an ESN, so lets create that using the reservoir and
    # input weights we built.
    esn = DiscreteEchoStateNetwork(reservoir, input_weights=input_weights,
                                   initial_state=dist,
                                   neuron_type='tanh',
                                   output_type='heaviside',
                                   output_neuron_pars={'shape': (mc_task.output_dimensions, 1),
                                                       'threshold': 0.0,
                                                       'newval': 1.0})

    # Then build the trainer, which defines our cost function
    trainer = LeastSquaredErrorTrainer()

    # The BenchmarkExperiment uses the ESN, task, and trainer
    test_exp = BenchmarkExperiment(esn, mc_task, trainer, num_training_trials=10,
                                   invert_target_of_training=False)
    test_exp.train_model()
    # Evaluate only runs one trial, put in a loop if you want to find an average
    print(test_exp.evaluate_model())


if __name__ == "__main__":
    test_experiment()
