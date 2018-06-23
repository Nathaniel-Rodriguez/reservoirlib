import numpy as np
from scipy import linalg
from abc import ABC


DEFAULT_FLOAT = np.float32
DEFAULT_INT = np.int64


class BaseActivator(ABC):
    """
    Defines the interface for the Activator class
    """

    @abstractmethod
    def __call__(self, x):
        """
        Should take a numpy array and return a reference to the same array.
        All operations should be in-place
        """
        pass

class Linear(BaseActivator):
    """
    Implements a linear functor that updates based on a linear function
    """
    def __init__(self, slope=1.0, bias=0.0, **kwargs):
        """
        y = slope * x + bias
        :param slope: scalar
        :param bias: scalar
        """
        super().__init__(**kwargs)

        self.slope = slope
        self.bias = bias

    def __call__(self, x):
        """
        Applies in-place linear activation
        :param x: numpy array
        :return: reference to x
        """

        np.multiply(x, self.slope, out=x)
        np.add(x, self.bias, out=x)
        return x


class Sigmoid(BaseActivator):
    """
    Implements a general sigmoid functor that updates a given numpy array
    in-place.
    """
    def __init__(self, a=1.0, b=1.0, c=0.0, d=0.0, e=1.0, **kwargs):
        """
        implements: a / (b + np.exp(-e * (x - c))) + d
        """
        super().__init__(**kwargs)

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e

    def __call__(self, x):
        """
        Applies sigmoid in-place
        :param x: A numpy array
        :return: reference to x
        """

        np.subtract(x, self.c, out=x)
        np.multiply(x, -self.e, out=x)
        np.exp(x, out=x)
        np.add(x, self.b, out=x)
        np.divide(self.a, x, out=x)
        np.add(x, self.d, out=x)

        return x


class InvertedSigmoid(Sigmoid):
    """
    Implements an inverted sigmoid using numpy arrays in-place
    """

    def __call__(self, x):
        """
        Applies inverted sigmoid in-place
        :param x: A numpy array
        :return: reference to x
        """

        np.subtract(x, self.d, out=x)
        np.divide(self.a, x, out=x)
        np.subtract(x, self.b, out=x)
        np.log(x, out=x)
        np.divide(x, -self.e, out=x)
        np.add(x, self.c, out=x)

        return x


class ArcTanh(BaseActivator):
    """
    Implements arctanh function. Just calls np.arctanh inplace on numpy array
    """
    def __call__(self, x):
        """
        in-place tanh operation
        :param x: a numpy array
        :return: reference to same array
        """
        np.arctanh(x, out=x)
        return x


class Tanh(BaseActivator):
    """
    Implements the tanh function. Just calls np.tanh inplace on numpy array
    """
    def __call__(self, x):
        """
        in-place tanh operation
        :param x: a numpy array
        :return: reference to same array
        """
        np.tanh(x, out=x)
        return x


class Identity:
    """
    Returns the same array
    """
    def __call__(self, x):
        """
        :param x: numpy array
        :return: returns reference to same array
        """
        return x


class Heaviside:
    """
    Implements the heaviside step function in-place.
    Requires allocating an array of the same shape as the one it is operating
    on in order to take advantage of c-speed comparisons in numpy.

    Returns an array with values that are either 0 or newval, depending on the
    threshold.
    """
    def __init__(self, shape, threshold=0.0, newval=1.0):
        """
        If below threshold, clamps to 0.
        If above, clamps to newval
        :param shape: shape of numpy array
        :param threshold: threshold between min/max value
        :param newval: clamped max value
        """
        self.bool_array = np.full(shape, True, dtype=bool)
        self.threshold = threshold
        self.newval = newval

    def __call__(self, x):
        """
        :param x: numpy array of shape == self.shape
        :return: reference to x
        """

        np.greater(x, self.threshold, out=self.bool_array)
        x[:] = 0
        np.putmask(x, self.bool_array, self.newval)

        return x


class BaseEchoStateNetwork(ABC):
    """
    Defines the interface for the EchoStateNetwork class
    """

    ActivationFunctions = {
        'sigmoid': Sigmoid,
        'invertedsigmoid': InvertedSigmoid,
        'arctanh': ArcTanh,
        'tanh': Tanh,
        'identity': Identity,
        'heaviside': Heaviside,
        'linear': Linear
    }

    @abstractmethod
    def reset(self):
        """
        Resets the state of the neural network, along with any other variables
        needed in order to do a fresh run again.
        """
        pass

    @abstractmethod
    def run(self, input_time_series=None, num_iter=None, record=False,
            output=False):
        """
        A method to run the neural network either from initial conditions or
        given some time-series.
        :param input_time_series: An optional input time-series
        :param num_iter: An optional number of iterations to run for
        :param record: Whether to record the history of the reservoir states
        :param output: Whether to calculate the output of the reservoir
        :return: output (if true)
        """
        pass

    @abstractmethod
    def train(self, input_time_series, target_output, *args, **kwargs):
        """
        sets the output weights of the ESN given # trials of input
        :param input_time_series: A list or array of input time-series
        :param target_output: A list or array of target time-series
        """
        pass


class DiscreteEchoStateNetwork(BaseEchoStateNetwork):
    """
    This is a discrete non-feedback ESN that uses linear regression learning.
    It can be expanded to include feedback, but this would require using a batch
    or online learning algorithm.

    The DiscreteEchoStateNetwork must be trained on input before an output weight
    matrix is generated.

    Available activation functions:
        sigmoid {a,b,c,d,e}
        invertedsigmoid {a,b,c,d,e}
        arctanh {}
        tanh {}
        identity {}
        heaviside {(num_neurons, 1), threshold, newval}

    Note: According to Jaeger, feedback is only needed for pattern generating ESNs.
    For time-series prediction, control, filtering, or pattern recognition,
    feedback is not advised.
    """

    def __init__(self, reservoir, input_weights=None, neuron_type="tanh", 
                 output_type="tanh", initial_state=None, neuron_pars=None,
                 output_neuron_pars=None, dtype=None, **kwargs):
        """
        :param reservoir: NxN numpy array
        :param input_weights: NxK numpy array, or None. Default: None
        :param neuron_type: string (tanh, sigmoid, heaviside, linear)
        :param output_type: string (tanh, sigmoid, heaviside, identity)
        :param initial_state: Distribution, Default: Zeros
            For Distribution class see reservoirgen
        :param neuron_pars: dict of parameters for neuron. Default: {}
        :param output_neuron_pars: dict of parameters for output neuron.
            Default: {}
        :param dtype: type of numpy arrays used. Default: DEFAULT_FLOAT

        Size key:
        N = # neurons in reservoir
        K = # input dimensions
        T = # of time steps
        C = # of time steps - cut
        S = time step stack of all trials: e.g. for 3 trials S = 3*C
        O = # of output dimensions
        """
        super().__init__(**kwargs)

        if dtype is None:
            self.dtype = DEFAULT_FLOAT
        if neuron_pars is None:
            neuron_pars = {}
        if output_neuron_pars is None:
            output_neuron_pars = {}

        if not isinstance(reservoir, np.ndarray):
            raise NotImplementedError("Reservoir needs to be a numpy array")

        if not (input_weights is None) and not isinstance(input_weights, np.ndarray):
            raise NotImplementedError("Input weights need to be a numpy "
                                      "array or None")

        # Weights
        self.dtype = dtype
        self.reservoir = reservoir  # NxN
        self.input_weights = input_weights  # NxK
        self.num_neurons = self.reservoir.shape[0]  # N
        if self.input_weights is None:
            self.num_inputs = 0
        else:
            self.num_inputs = self.input_weights.shape[1]  # K
        self.initial_state = initial_state

        # Set neuron types (reservoir)
        self.neuron_type = neuron_type
        self.neuron_pars = neuron_pars
        self.activation_function = DiscreteEchoStateNetwork.ActivationFunctions[
            neuron_type.lower()](**neuron_pars)

        # Set neuron types (output neuron)
        self.output_type = output_type
        self.output_neuron_pars = output_neuron_pars
        self.output_function = DiscreteEchoStateNetwork.ActivationFunctions[
            output_type.lower()](**output_neuron_pars)

        # Initialize system, except for history
        self.iteration = 0

        # An Nx1 array
        self.input_state = np.zeros((self.num_neurons, 1), dtype=self.dtype)

        # An Nx1 array
        self.state = self.generate_initial_state(np.zeros((self.num_neurons, 1),
                                                          dtype=self.dtype))

        # Will be a TxNx1 array
        self.history = None

        # (N+K)xO trained weight matrix
        self.output_weight_matrix = None
        # transpose of the above: Ox(N+K)
        self.output_weight_matrix_t = None

        # (N+K)x1 state concatenated with inputs
        self.full_state = np.zeros((self.num_neurons + self.num_inputs, 1),
                                   dtype=self.dtype)

        # Output record will be an TxO array
        self.output = None

    def generate_initial_state(self, x):
        """
        Sets all initial states
        self.initial_state should be either None, or a Distribution or other
        callable object that can be given a size argument
        """

        if self.initial_state is None:
            x[:] = 0
            return x
        else:
            x[:] = self.initial_state(size=(self.num_neurons, 1))
            return x

    def reset(self):
        """
        :return: None
        """
        self.iteration = 0
        self.state = self.generate_initial_state(self.state)
        self.input_state[:] = 0.0
        self.full_state[:] = 0.0

    def step(self, input_array, record=False):
        """
        Ideally the view should be over the major axis, which for numpy is rows.
        :param input_array: a Kx1 numpy array
        """

        np.dot(self.input_weights, input_array, out=self.input_state)
        np.dot(self.reservoir, self.state, out=self.state)
        np.add(self.input_state, self.state, out=self.state)
        self.activation_function(self.state)

        if record:  # Assigns values from current state to history
            self.history[self.iteration][:] = self.state[:]

    def no_input_step(self, record=False):
        """
        A stepping function that doesn't require input. Just calls the reservoir
        on itself.
        """

        np.dot(self.reservoir, self.state, out=self.state)
        self.activation_function(self.state)

        if record:  # Assigns values from current state to history
            self.history[self.iteration][:] = self.state[:]

    def response(self, input_array):
        """
        Calculate the networks response given an input_array view
        :param input_array: a Kx1 numpy array
        :return: Ox1 numpy array
        """

        self.full_state[:self.num_neurons] = self.state
        self.full_state[self.num_neurons:] = input_array
        # Ox(N+K) * (N+k)x1
        np.dot(self.output_weight_matrix_t, self.full_state, out=self.full_state)
        return self.output_function(self.full_state)

    def run(self, input_time_series=None, num_iter=None, record=False,
            output=False):
        """
        Re-initializes the history
        Re-initializes the output

        :param input_time_series: Input time series, should be numpy array with
            shape TxKx1
        :param num_iter: If not None, runs till number of iterations
        :param record: toggle whether to record state history. Defaulted to record
            attribute. Setting this changes the record attribute's value.
        :param output: If True, the response is evaluated and returned
        :return: the output record: TxO if output=True, else None

        :note: Response is not evaluated when no time-series is given, just a
            zeroed array is returned
        """

        if num_iter is None:
            num_iter = len(input_time_series)

        # Initialize the history for this run
        if record:
            self.history = np.zeros((num_iter, self.num_neurons, 1),
                                    dtype=self.dtype)
            
        # If not time-series, run step without input
        if input_time_series is None:
            for i in range(num_iter):
                self.iteration = i
                self.no_input_step(record=record)

        # Evaluate for time series input
        else:
            # Initialize the output for this run
            if hasattr(self, "output_weight_matrix_t"):
                # Has shape TxO
                self.output = np.zeros((num_iter,
                                        self.output_weight_matrix_t.shape(0)),
                                       dtype=self.dtype)

            for i in range(num_iter):
                self.iteration = i
                self.step(input_time_series[i], record=record)
                if output:
                    self.output[i, :] = np.squeeze(self.response(input_time_series[i]),
                                                   axis=1)

        if output:
            return self.output

    def invert_target_array(self, target_output):
        """
        Creates a copy of the target_output that will be inverted
        :param target_output: a numpy array
        """

        target_output = target_output.copy()
        if self.output_type == 'sigmoid':
            return InvertedSigmoid(**self.output_neuron_pars)(target_output)

        elif self.output_type == 'tanh':
            return ArcTanh()(target_output)

    def set_output_weights(self, weight_matrix):
        """
        Takes the weights determined by the regression and assigned them
        to the output weight matrix. It also stores the transpose for
        evaluating the network response.
        """

        self.output_weight_matrix = weight_matrix
        # NOTE: transpose returns a view, which is undesirable since it
        # doesn't change underlying memory position of elements, harming
        # performance. Using copy() defaults the resulting array to C-order
        # memory layout.
        self.output_weight_matrix_t = weight_matrix.transpose().copy()

    def train(self, input_time_series, target_output, cuts=None, invert_target=False,
              cut_output=False):
        """
        Trains the agent using either an single time series input or multiple
        time-series.
        
        :param input_time_series: a list of numpy arrays with shape QxTxKx1 or TxKx1.
            Where Q=#trials, T=time, K=#inputs
            If it has 4 dimensions, multi-trial training is used.
            If it has 3 dimensions, single trial training is used.
            If input_time_series is a list, it is assumed that it is a list
            of arrays with shape TxKx1.
        :param target_output: a numpy array with shape QxTxOx1 or TxOx1.
            Where O=#outputs.
            Dimension should match input time series.
        :param cuts: Removes first cuts time steps from reservoir history from
            training evaluation. cuts can either be a sequence or scalar value.
            If multi-trial is used, then cuts is assumed to be a sequence, a cut
            for each trial.
            If single-trial is used, then cuts is assumed to be a scalar.
            If None, no cut is used.
            Default: None
        :param invert_target: Specifies whether to invert the target values for
            training. Maybe necessary for tanh and sigmoid functions to convert
            to a linear space for training. Default: False
        :param cut_output: true/false, determine whether to apply the cuts on the
            input time-series to the target time-series
        :return: None

        TODO: Make work with list of time-series
        TODO: Make work with variable time-series lengths
        TODO: Make work with AxB input array instead of AxBx1 by doing it internally
        """

        num_trials = input_time_series.shape[0]

        # Carry out inversion if option is selected
        if invert_target:
            target_output = self.invert_target_array(target_output)

        # Check shapes. If 3 dimensions are used, expand array by a single
        # dimension. This is so that it can be handled as if it were 4-dims
        if (len(input_time_series.shape) == 3) and (len(target_output.shape) == 3):
            input_time_series = np.extend_dims(input_time_series, axis=0)  # new view
            target_output = np.extend_dims(target_output, axis=0)  # new view
            if cuts is None:
                cuts = [0]
            else:
                cuts = [cuts]

        elif (len(input_time_series.shape) == 4) and (len(target_output.shape) == 4):
            if cuts is None:
                cuts = [0 for i in range(num_trials)]

        else:
            raise NotImplementedError("input and target must have same number"
                                      " of dimensions (either 3 or 4)")

        if cut_output:
            output_cut = cuts
        else:
            output_cut = [0 for i in range(num_trials)]

        # This is time-series length following the cut and after stacking
        stacked_time_series_length = np.sum([input_time_series[i].shape[0] - cuts[i]
                                             for i in range(num_trials)])

        # determines the stack index, for assigning output of each trial
        # to the proper place in the full stacked matrix
        def index(x, k, c):
            return int(np.sum([x[v].shape[0] - c[v] for v in range(k)]))

        # With cut introduced, the stacked full history is a Sx(N+K) matrix
        # where S=stacked length
        stacked_full_history = np.zeros((stacked_time_series_length,
                                         self.num_inputs + self.num_neurons),
                                        dtype=self.dtype)
        # Go through each trail, generate output and stack the C length axis onto the S
        # axis of the stacked array
        for trial_num in range(num_trials):
            # make sure recording is set to True, necessary for training and run trial
            self.run(input_time_series[trial_num], record=True)

            # Cut history and fill the full history
            cut_history = self.history[cuts[trial_num]:]  # CxNx1
            stacked_full_history[index(input_time_series, trial_num, cuts):
                                 index(input_time_series, trial_num + 1, cuts),
                                 : cut_history.shape(1)] = np.squeeze(cut_history,
                                                                      axis=2)

            # Cut input time series for this trial and fill full history
            cut_input_time_series = input_time_series[trial_num][cuts[trial_num]:]  # CxKx1
            stacked_full_history[index(input_time_series, trial_num, cuts):
                                 index(input_time_series, trial_num + 1, cuts),
                                 cut_history.shape(1):] = np.squeeze(cut_input_time_series,
                                                                     axis=2)

            # After trial is complete the simulation needs to be reset
            self.reset()

        # Stack target output:
        num_outputs = target_output[0].shape[1]
        stacked_target_output = np.zeros((stacked_time_series_length, num_outputs),
                                         dtype=self.dtype)  # SxO
        for trial_num in range(num_trials):
            cut_target_output = target_output[trial_num][output_cut[trial_num]:]  # CxOx1
            stacked_target_output[index(target_output, trial_num, output_cut):
                                  index(target_output, trial_num + 1, output_cut),
                                  :] = np.squeeze(cut_target_output, axis=2)

        # Run regression and then set the ESN output weights
        # input: Sx(N+K) and SxO
        # solution: (N+K)xO
        solution, residuals, rank, sing = linalg.lstsq(stacked_full_history,
                                                       stacked_target_output)
        self.set_output_weights(solution)

    def predict(self, input_time_series, target_output, cut=0,
                target_range=None, error_type='NRMSE', analysis_mode=False):
        """
        Evaluates the performance of a trained ESN on a given input trial.
        T=time steps
        K=# inputs
        O=# outputs

        :param input_time_series: a TxKx1 numpy array
        :param target_output: a TxOx1 or TxO numpy array
        :param cut: scalar, number of initial time-steps to drop
        :param target_range: tuple (target min value, target max value)
            Used for normalizing the RMSE. Default: Uses target_output min/max
        :param error_type: 'NRMSE' or 'AE' (absolute error)
        :param analysis_mode: T/F Specifies whether to return all outputs.
            Default: False
        :return: performance
            or performance, cut_prediction, target_output, prediction
        """

        # TxO
        prediction = self.run(input_time_series, record=analysis_mode,
                              output=True)

        cut_prediction = prediction[cut:]  # CxO

        cut_target_output = target_output[cut:]  # CxOx1
        cut_target_output = np.squeeze(cut_target_output, axis=1)  # CxO
        residuals = np.abs(cut_prediction - cut_target_output)  # CxO

        performance = None
        if error_type == 'NRMSE':
            if not (target_range is None):
                min_target = target_range[0]
                max_target = target_range[1]
            else:
                min_target = np.min(cut_target_output)
                max_target = np.max(cut_target_output)

            performance = normalize_root_mean_squared_error(residuals,
                                                            max_target,
                                                            min_target)

        elif error_type == 'AE':
            performance = absolute_error(residuals)

        if analysis_mode:
            return performance, cut_prediction, target_output, prediction
        else:
            return performance


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


if __name__ == '__main__':
    """
    testing
    """

    pass
