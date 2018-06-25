import numpy as np
from reservoirlib.esn import ArcTanh
from reservoirlib.esn import InvertedSigmoid


class BenchmarkExperiment:
    """
    Provides a framework for running an experiment using various tasks
    """

    def __init__(self, esn, task, trainer, num_training_trials,
                 invert_target_of_training=False):
        """
        :param esn: an object that can be trained on task data
            esn should conform to the BaseESN interface
        :param task: an object that will generate input and target time-series
            task should conform the BaseTask interface
        :param trainer: a function that takes as arguments
            a target and model numpy array and returns a set of parameters
        :param num_training_trials: number of trials to run the task
        :param invert_target_of_training: whether to invert the target signal
            for training. This is usually True if ESN output layer has tanh or
            sigmoid neurons and you are using least squared regression. Else
            keep false. Default: False
        """

        self.esn = esn
        self.task = task
        self.trainer = trainer
        self.num_training_trials = num_training_trials
        self.invert_target_of_training = invert_target_of_training

    def train_model(self):
        """
        Generates input and target signals and runs the ESN's training algorithm
        :return: None
        """
        # Generate data for training
        input_trials = [None for i in range(self.num_training_trials)]
        target_trials = [None for i in range(self.num_training_trials)]
        history_trials = [None for i in range(self.num_training_trials)]
        input_cuts = [None for i in range(self.num_training_trials)]
        output_cuts = [None for i in range(self.num_training_trials)]
        for i in range(self.num_training_trials):
            input_trials[i], target_trials[i], input_cuts[i], output_cuts[i] = \
                self.task.generate_signal()
            input_trials[i] = np.expand_dims(input_trials[i], axis=2)
            target_trials[i] = np.expand_dims(target_trials[i], axis=2)
            self.esn.run(input_trials[i], record=True, output=False)
            history_trials[i] = np.zeros((input_trials[i].shape[0],
                                          self.esn.num_neurons, 1),
                                         dtype=self.esn.dtype)
            history_trials[i][:] = self.esn.history
            self.esn.reset()

        # stack trial data
        stacked_target = stack_targets(target_trials, output_cuts, self.esn.dtype)
        stacked_history = stack_history(input_trials, history_trials, input_cuts,
                                        self.esn.dtype)

        # invert target stack data if applicable
        if self.invert_target_of_training:
            invert_target_array(stacked_target, self.esn.output_type,
                                self.esn.output_neuron_pars)

        # train on data
        solution = self.trainer(stacked_history, stacked_target)

        # call set weights for esn
        self.esn.set_output_weights(solution)

    def evaluate_model(self):
        """
        Task specific validation runs
        :return: The task specific validation output
        """

        input_signal, target_output, in_cut, out_cut = self.task.generate_signal()
        prediction = self.esn.run(np.expand_dims(input_signal, axis=2),
                                  output=True)
        return self.task.validate(prediction, target_output)


class DynamicsExperiment:
    """
    Provides a framework for evaluating dynamical attributes of a neural
    network.
    """

    def __init__(self, esn, task, metric):
        """
        :param esn: an object that should conform to the BaseESN interface
        :param task: an object that will generate input time-series
            task should conform the BaseTask interface
        :param metric: a function that takes as arguments
            the history of an ESN and evaluates. Should conform to BaseMetric
            interface.
        """

        self.esn = esn
        self.task = task
        self.metric = metric

    def evaluate_metric(self):
        """
        Run the ESN on the task
        :return: the metric evaluated on the history of the ESN
        """

        input_signal, _, _, _ = self.task.generate_signal()
        self.esn.run(np.expand_dims(input_signal, axis=2), record=True, output=False)
        return self.metric(np.squeeze(self.esn.history, axis=2))


def index(x, k, c):
    """
    Determines the stack index, for assigning output of each trial to the proper
    place in the full stacked matrix
    :param x: time-series array
    :param k: trial #
    :param c: cuts
    :return: index
    """
    return int(np.sum([x[v].shape[0] - c[v] for v in range(k)]))


def stack_targets(target_outputs, cuts, dtype):
    """
    Stackes the multiple numpy arrays for each trial into a new array.
    :param target_outputs: a list of numpy arrays with shape TxOx1.
        Shape of data structure is QxTxOx1.
        Where O=#outputs Q=#trials T=time (can be variable)
        S = time step stack of all trials: e.g. for 3 trials S = 3*C
    :param cuts: sequence of cuts, one for each trial
    :param dtype: numpy type for stacked array
    :return: a numpy array of stacked target output with shape SxO
    """
    num_trials = len(target_outputs)

    # Stack target output:
    num_outputs = target_outputs[0].shape[1]
    # This is target length following the cut and after stacking
    stacked_target_length = np.sum([target_outputs[i].shape[0] - cuts[i]
                                    for i in range(num_trials)])
    stacked_target_output = np.zeros((stacked_target_length, num_outputs),
                                     dtype=dtype)  # SxO
    for trial_num in range(num_trials):
        stacked_target_output[index(target_outputs, trial_num, cuts):
                              index(target_outputs, trial_num + 1, cuts),
        :] = np.squeeze(target_outputs[trial_num][cuts[trial_num]:], axis=2)

    return stacked_target_output


def stack_history(inputs, histories, cuts, dtype):
    """

    :param inputs: a list of numpy arrays with shape TxKx1. Data structure shape
        is QxTxKx1.
        Where Q=#trials, T=time (can be variable), K=#inputs
        S = time step stack of all trials: e.g. for 3 trials S = 3*C
    :param histories: a list of numpy array with shape TxNx1.
    :param cuts: Removes first cuts time steps from reservoir history from
        training evaluation.
    :param num_inputs: number of input dimensions
    :param dtype: type or returned stacked array
    :return: a stacked array with shape Sx(N+K)
    """
    num_trials = len(inputs)
    num_neurons = histories[0].shape[1]
    num_inputs = inputs[0].shape[1]

    # This is time-series length following the cut and after stacking
    stacked_inputs_length = np.sum([inputs[i].shape[0] - cuts[i]
                                    for i in range(num_trials)])

    # With cut introduced, the stacked full history is a Sx(N+K) matrix
    # where S=stacked length
    stacked_history = np.zeros((stacked_inputs_length, num_inputs + num_neurons),
                               dtype=dtype)

    # Go through each trail, generate output and stack the C length axis onto the S
    # axis of the stacked array
    for trial_num in range(num_trials):

        # Cut history and fill the full history
        cut_history = histories[trial_num][cuts[trial_num]:]  # CxNx1
        stacked_history[index(histories, trial_num, cuts):
                        index(histories, trial_num + 1, cuts),
                        : cut_history.shape[1]] = np.squeeze(cut_history, axis=2)

        # Cut input time series for this trial and fill full history
        cut_inputs = inputs[trial_num][cuts[trial_num]:]  # CxKx1
        stacked_history[index(inputs, trial_num, cuts):
                        index(inputs, trial_num + 1, cuts),
                        cut_history.shape[1]:] = np.squeeze(cut_inputs, axis=2)

    return stacked_history


def invert_target_array(target_output, output_type, output_neuron_pars):
    """
    Creates a copy of the target_output that will be inverted
    :param target_output: a numpy array
    :param output_type: a string designating activation type
    :param output_neuron_pars: dictionary of parameters for activation type
    """

    target_output = target_output.copy()
    if output_type == 'sigmoid':
        return InvertedSigmoid(**output_neuron_pars)(target_output)

    elif output_type == 'tanh':
        return ArcTanh()(target_output)

    else:
        raise NotImplementedError("Inversion for this activation function is"
                                  " not supported.")
