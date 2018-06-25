import numpy as np
from abc import ABC, abstractmethod
from reservoirlib.utilities import DEFAULT_FLOAT
from reservoirlib.utilities import normalize_root_mean_squared_error
from reservoirlib.utilities import absolute_error


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

    @property
    @abstractmethod
    def output_type(self):
        """
        :return: string, output layer neuron type
        """
        pass

    @output_type.setter
    def output_type(self, val):
        raise NotImplementedError("Cant change neuron output types")

    @property
    @abstractmethod
    def output_neuron_pars(self):
        """
        :return: string, output layer neuron parameters
        """
        pass

    @output_neuron_pars.setter
    def output_neuron_pars(self, val):
        raise NotImplementedError("Cant change neuron output parameters")

    @property
    @abstractmethod
    def num_neurons(self):
        """
        :return: number of neurons in ESN
        """
        pass

    @num_neurons.setter
    def num_neurons(self, val):
        raise NotImplementedError("Cant change # neurons")
    
    @property
    @abstractmethod
    def dtype(self):
        """
        :return: returns type of numpy arrays the ESN uses 
        """
        pass
    
    @dtype.setter
    def dtype(self, val):
        raise NotImplementedError("Can't change dtype")

    @property
    @abstractmethod
    def history(self):
        """
        :return: returns reference to ESN state history 
        """
        pass
    
    @history.setter
    def history(self, val):
        raise NotImplementedError("Not allowed to set history")

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
    def set_output_weights(self, weight_matrix):
        """
        Takes the weights determined by the regression and assigned them
        to the ESNs output layer.
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
            self._dtype = DEFAULT_FLOAT
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
        self._dtype = dtype
        self.reservoir = reservoir  # NxN
        self.input_weights = input_weights  # NxK
        self._num_neurons = self.reservoir.shape[0]  # N
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
        self._output_type = output_type
        self._output_neuron_pars = output_neuron_pars
        self.output_function = DiscreteEchoStateNetwork.ActivationFunctions[
            output_type.lower()](**output_neuron_pars)

        # Initialize system, except for history
        self.iteration = 0

        # An Nx1 array
        self.input_state = np.zeros((self._num_neurons, 1), dtype=self._dtype)

        # An Nx1 array
        self.state = self.generate_initial_state(np.zeros((self._num_neurons, 1),
                                                          dtype=self._dtype))

        # Will be a TxNx1 array
        self._history = None

        # (N+K)xO trained weight matrix
        self.output_weight_matrix = None
        # transpose of the above: Ox(N+K)
        self.output_weight_matrix_t = None

        # (N+K)x1 state concatenated with inputs
        self.full_state = np.zeros((self._num_neurons + self.num_inputs, 1),
                                   dtype=self._dtype)

        # Output record will be an TxO array
        self.output = None

    @property
    def output_type(self):
        """
        :return: string, type of output neurons
        """
        return self._output_type

    @property
    def output_neuron_pars(self):
        """
        :return: parameters for output neurons
        """
        return self._output_neuron_pars

    @property
    def num_neurons(self):
        """
        :return: number of neurons in ESN
        """
        return self._num_neurons

    @property
    def history(self):
        """
        :return: returns reference to ESN state history 
        """
        return self._history

    @property
    def dtype(self):
        """
        :return: returns type of numpy arrays the ESN uses 
        """
        return self._dtype

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
            x[:] = self.initial_state(size=(self._num_neurons, 1))
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
        :param record: whether to record state in history
        """

        np.dot(self.input_weights, input_array, out=self.input_state)
        np.dot(self.reservoir, self.state, out=self.state)
        np.add(self.input_state, self.state, out=self.state)
        self.activation_function(self.state)

        if record:  # Assigns values from current state to history
            self._history[self.iteration][:] = self.state[:]

    def no_input_step(self, record=False):
        """
        A stepping function that doesn't require input. Just calls the reservoir
        on itself.
        """

        np.dot(self.reservoir, self.state, out=self.state)
        self.activation_function(self.state)

        if record:  # Assigns values from current state to history
            self._history[self.iteration][:] = self.state[:]

    def response(self, input_array):
        """
        Calculate the networks response given an input_array view
        :param input_array: a Kx1 numpy array
        :return: Ox1 numpy array
        """

        self.full_state[:self._num_neurons] = self.state
        self.full_state[self._num_neurons:] = input_array
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
            self._history = np.zeros((num_iter, self._num_neurons, 1),
                                     dtype=self._dtype)
            
        # If not time-series, run step without input
        if input_time_series is None:
            for i in range(num_iter):
                self.iteration = i
                self.no_input_step(record=record)

        # Evaluate for time series input
        else:
            if output:
                # Initialize the output for this run
                self._initialize_output(num_iter)

            for i in range(num_iter):
                self.iteration = i
                self.step(input_time_series[i], record=record)

                if output:
                    self.output[i, :] = np.squeeze(self.response(input_time_series[i]),
                                                   axis=1)

        if output:
            return self.output

    def _initialize_output(self, num_iter):
        """
        Only allocates memory if current array doesn't exist or isn't the
        right shape.
        :return: output array with shape TxO
        """

        if hasattr(self, "output_weight_matrix_t") and hasattr(self, "output"):
            if self.output.shape[0] == num_iter:
                return self.output

        else:
            # Has shape TxO
            return np.zeros((num_iter,
                             self.output_weight_matrix_t.shape[0]),
                            dtype=self._dtype)

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


if __name__ == '__main__':
    """
    testing
    """

    pass
