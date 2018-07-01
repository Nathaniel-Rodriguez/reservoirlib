from reservoirlib.esn import DiscreteEchoStateNetwork
from reservoirlib.task import BinaryMemoryCapacityTask
from reservoirlib.experiment import BenchmarkExperiment
from reservoirlib.trainer import LeastSquaredErrorTrainer
from reservoirlib.distribution import Distribution
from reservoirlib.generator import generate_reservoir_input_weights
from reservoirlib.generator import generate_adj_reservoir_from_adj_matrix
from graphgen.lfr_generators import unweighted_directed_lfr_as_adj
import numpy as np

# This import requires mpi4py module to be installed, which means you need
# a working MPI, like mpich or openMPI or Cray's Aprun (craype).
from reservoirlib.utility import run_mpi_experiment


def mc_task_example(parameters):
    """
    :param parameters: a set of parameters that you wish to vary or that need
        to vary from experiment to experiment (such as seeds or control variables)
    :return: memory capacity performance on the mc task for the reservoir
    """
    input_weight_dist = Distribution("uniform", {'low': -1.0, 'high': 1.0},
                                     seed=parameters['input_weight_seed'])
    reservoir_dist = Distribution("uniform", {'low': -1.0,
                                              'high': 1.0},
                                  seed=parameters['reservoir_seed'])

    # Construct graph
    graph_pars = {'num_nodes': 500,
                  'average_k': 6,
                  'max_degree': 6,
                  'mu': parameters['mu'],
                  'com_size_min': 10,
                  'com_size_max': 10,
                  'seed': parameters['graph_seed'],
                  'transpose': True}
    graph, _ = unweighted_directed_lfr_as_adj(**graph_pars)

    # construct task
    mc_task = BinaryMemoryCapacityTask(duration=2000, cut=100, num_lags=200, shift=1)

    # construct reservoir
    input_weights = generate_reservoir_input_weights(mc_task.input_dimensions,
                                                     graph_pars['num_nodes'],
                                                     parameters['r_sig'],
                                                     input_weight_dist,
                                                     by_dimension=True)
    reservoir = generate_adj_reservoir_from_adj_matrix(graph, reservoir_dist)

    # Adjust for spectral radius
    spectral_radius = np.sort(np.absolute(np.linalg.eigvals(reservoir)))[-1]
    np.divide(reservoir, spectral_radius, out=reservoir)
    np.multiply(reservoir, parameters['scale'], out=reservoir)

    # Construct ESN
    esn = DiscreteEchoStateNetwork(reservoir, input_weights=input_weights,
                                   initial_state=None,
                                   neuron_type='linear',
                                   neuron_pars={'slope': 1.0, 'bias': 0.0},
                                   output_type='heaviside',
                                   output_neuron_pars={'shape': (mc_task.output_dimensions, 1),
                                                       'threshold': 0.5, 'newval': 1.0})
    trainer = LeastSquaredErrorTrainer()
    test_exp = BenchmarkExperiment(esn, mc_task, trainer, num_training_trials=1,
                                   invert_target_of_training=False)
    test_exp.train_model()
    # Evaluate only runs one trial, put in a loop if you want to find an average
    mc, _, _ = test_exp.evaluate_model()

    return mc


if __name__ == "__main__":

    """
    The function mc_task_example takes a set of parameters as an argument and
    instantiates and runs the experiment. The run_mpi_experiment function
    can take a function and a list of parameters as an argument, and it will
    distribution experimental runs over multiple machines. It automatically
    divides the experiments between the nodes and return the results in the
    order they were given.
    
    If the use wants to, say, vary two parameters at once, then can make a flat
    list of parameters and then after running run_mpi_experiment can unpack
    or reshape the list.
    
    By default no results are returned from the function, instead they are pickled
    and dumped to file at the end. 
    """

    # First define parameters for each experiment
    mus = np.linspace(0.0, 0.5, 20)
    r_sigs = np.linspace(0.05, 0.6, 20)
    num_trials = 50
    rng = np.random.RandomState(765)
    parameters = [{'input_weight_seed': rng.randint(0, 1000000),
                   'reservoir_seed': rng.randint(0, 1000000),
                   'mu': mu,
                   'graph_seed': rng.randint(0, 1000000),
                   'r_sig': r_sig,
                   'scale': 0.9}
                  for mu in mus
                  for r_sig in r_sigs
                  for trial in range(num_trials)]

    # Use run_mpi function to distribution over multiple nodes
    run_mpi_experiment(mc_task_example, parameters,
                       savefile='example_results.pyobj',
                       return_results=False)
