import numpy as np
from abc import ABC


class BaseTask(ABC):
    """
    Abstract Base class for discrete tasks
    """

    def __init__(self, dtype=None):
        """
        :param dtype: the output type of the input/target arrays
            Default: np.float64
        """

        if dtype is None:
            self.dtype = np.float64
        else:
            self.dtype = dtype

    @property
    @abstractmethod
    def input_dimensions(self):
        """
        Defines a tuple/sequence of the input shape (not including time)
        :return: returns shape
        """
        pass

    @input_dimensions.setter
    def input_dimensions(self, val):
        raise NotImplementedError("Can not set this input_dimensions")

    @property
    @abstractmethod
    def output_dimensions(self):
        """
        Defines a tuple/sequence of the output shape (not including time)
        :return: returns shape
        """
        pass

    @output_dimensions.setter
    def output_dimensions(self, val):
        raise NotImplementedError("Can not set output_dimensions")

    @abstractmethod
    def generate_signal(self):
        """
        :param num_steps: duration of signal (if applicable
        :return: a tuple with first array being the input signal, and second
            array being the target output. Uses a numpy array with the signal,
            Time will always be the most minor dimension and it can be of
            variable length. The input and output arrays don't have to be
            of the same length or dimension.
            The tuples should also contain the input and output cuts respectively.
            This is the number of time-steps to drop from training for a given
            generated sequence of both the input series and the target series.
        """
        pass

    @abstractmethod
    def validate(self, prediction, target):
        """
        Uses the model prediction and target to evaluate the performance of
        the model
        :param prediction: a numpy array of model prediction
        :param target: a numpy array of target output
        :return: a set of relevant performance statistics
        """
        pass


class NbitRecallTask(BaseTask):
    """
    Implements the recall task used in our paper.
    A pattern is generated, followed by a distractor, followed by a cue, after
    which a learner must reproduce the pattern.
    """

    def __init__(self, pattern_length=1, pattern_dimension=1, start_time=0,
                 distraction_duration=0, cue_value=1, distractor_value=0,
                 pattern_value=1, loop_unique_input=False, seed=None, **kwargs):
        """
        Initializes the task. The Pattern is a set of values that activate
        for each time-step for a set number of time-steps. Only one dimension of
        the pattern activates per time-step. This is done to ensure total
        input activity remains constant.
        :param pattern_length: the number of time-steps the pattern activates for
        :param pattern_dimension: the number of possible activation dimensions
        :param start_time: when the pattern starts getting emitted
        :param distraction_duration: how long after the pattern is complete until
            the cue prompts recall.
        :param cue_value: the input value of the cue
        :param distractor_value: the input value of the distractor
        :param pattern_value: the value of the pattern activation
        :param loop_unique_input: if True, all pattern configurations are
            generated and those are used for the signals. If False, random
            patterns are generated.
        :param seed: for the RNG, if None (default), then it uses numpy
            RandomState's default seed
        :param kwargs: BaseTask arguments
        """
        super().__init__(**kwargs)

        self._seed = seed
        if seed is None:
            self._rng = np.random.RandomState()
        else:
            self._rng = np.random.RandomState(self._seed)

        self.pattern_length = pattern_length
        self.pattern_dimension = pattern_dimension
        self.start_time = start_time
        self.distraction_duration = distraction_duration
        self.cue_value = cue_value
        self.distractor_value = distractor_value
        self.pattern_value = pattern_value
        self.loop_unique_input = loop_unique_input
        self.cue_time = self.start_time + self.pattern_length + self.distraction_duration
        self.recall_time = self.cue_time + 1
        self.distractor_state_time = self.start_time + self.pattern_length
        self.total_duration = self.start_time + self.pattern_length + \
                              self.distraction_duration + 1 + self.pattern_length
        if self.loop_unique_input:
            self.looping_index = 0  # used for drawing from shuffled patterns
            self.pre_generate_all_patterns()

        self._input_dimensions = self.pattern_dimension + 2  # + cue + distractor
        self._output_dimensions = self.pattern_dimension

    @property
    def input_dimensions(self):
        return self._input_dimensions

    @property
    def output_dimensions(self):
        return self._output_dimensions

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        self._rng.seed(self._seed)

    def pre_generate_all_patterns(self):
        """
        Advise to only use for small dimensions
        """
        import itertools

        full_index_set = list(itertools.product(range(self.pattern_dimension),
                                                repeat=self.pattern_length))

        self.pregenerated_patterns = np.zeros((len(full_index_set),
                                               self.pattern_length,
                                               self.pattern_dimension),
                                              dtype=self.dtype)
        for i, index_set in enumerate(full_index_set):
            for j in range(self.pattern_length):
                self.pregenerated_patterns[i, j, index_set[j]] = 1

        self._rng.shuffle(full_index_set)  # only suffles along first axis

    def draw_random_pattern(self, target_output):
        """
        uses RNG to generate a random pattern
        :param target_output: in-place modification of target_output
        :return: reference to target_output
        """
        # Create sequence to memorize
        activation_indices = self._rng.choice(range(self.pattern_dimension),
                                              size=self.pattern_length,
                                              replace=True)
        for i in range(self.pattern_length):
            target_output[i, activation_indices[i]] = self.pattern_value

        return target_output

    def draw_pregenerated_pattern(self, target_output):
        """
        uses pregenerated patterns
        :param target_output: in-place modification of target_output
        :return: reference to target_output
        """

        target_output[:] = self.pregenerated_patterns[self.looping_index]
        self.looping_index = (self.looping_index + 1) \
                             % self.pregenerated_patterns.shape[0]

        return target_output

    def draw_pattern(self):
        """
        Draw either a random pattern or a pre-generated one.
        :return: target output, a Pl x Pd numpy array
        """

        target_output = np.zeros((self.pattern_length, self.pattern_dimension),
                                 dtype=self.dtype)

        if self.loop_unique_input:
            return self.draw_pregenerated_pattern(target_output)
        else:
            return self.draw_random_pattern(target_output)

    def generate_signal(self):
        """
        Creates a pattern randomly.
        Generates a time-series and a target time-series.
        The time-series is of dimensions (T+2Pl+1) x (Pd + 2)
        The target is of dimensions (Pl) X (Pd)
        Pl is the pattern length.
        Pd is the pattern dimension.
        T is the time duration of the distractor.
        recall time = distractor duration + Pl + start time.
        :return: Returns the input array and target output array
        """

        input_signal = np.zeros((self.total_duration, self._input_dimensions),
                                dtype=self.dtype)
        target_output = self.draw_pattern()

        # Set input signal pattern
        input_signal[self.start_time:self.start_time + self.pattern_length,
        :self.pattern_dimension] = target_output
        # Create distractor series (it is the second to last dimension)
        input_signal[self.distractor_state_time:
                     self.distractor_state_time + self.distraction_duration,
                     :-2] = self.distractor_value
        # Create cue series (it is the last dimension)
        input_signal[self.cue_time: -1] = self.cue_value

        return input_signal, target_output

    def validate(self, prediction, target):
        """
        :param prediction: full time-series response of model
        :param target: target pattern
        :return: 1 if correctly identified or 0 if not
        """

        prediction_pattern = prediction[self.recall_time:, :-2]
        rounded_prediction = np.rint(prediction_pattern).astype(np.int64)
        rounded_target = np.rint(target).astype(np.int64)
        if np.sum(rounded_target - rounded_prediction) == 0:
            return 1
        else:
            return 0


class MemoryCapacityTask(BaseTask):
    """
    Implements the MC task used by Jaeger (pretty close to it I believe)
    """

    def __init__(self, duration, activation_distribution, cut=0, num_lags=1, shift=1, **kwargs):
        """
        :param duration: the number of trainable time-steps to generate the
            signal for.
            Input signal duration will be: cut + duration + num_lags*shift
            Target signal will be: duration

        :param activation_distribution: a Distribution from which input values
            will be drawn. Should be callable with a size argument for drawing
            random values.
        :param cut: the number of time-steps to drop from target
        :param num_lags: the number of lags to use for creating target output
        :param shift: how many lags to skip, e.g. if shift=1, then lags are
            1,2,3... if shift=2, then lags are 1,3,5...
        :param kwargs: BaseTask parameters
        """
        super().__init__(**kwargs)

        self.duration = duration
        self.activation_distribution = activation_distribution
        self.cut = cut
        self.num_lags = num_lags
        self.shift = shift
        self.max_lag = num_lags * shift

        self.input_dimensions = 1
        self.output_dimensions = num_lags

    @property
    def input_dimensions(self):
        return self._input_dimensions

    @property
    def output_dimensions(self):
        return self._output_dimensions

    def generate_input_signal(self):
        """
        :return: the input signal of the task
        """

        return self.activation_distribution((self.cut + self.duration
                                             + self.max_lag, 1)
                                            ).astype(dtype=self.dtype)

    def generate_signal(self):
        """
        :return: input with shape (cut + duration + num_lags*shift) x (1)
            target output with shape (duration) x (num_lags)
        """

        input_signal = self.generate_input_signal()

        target_signal = np.zeros((self.duration, self.num_lags), dtype=self.dtype)
        for i in range(target_signal.shape[1]):
            target_signal[:, i] = input_signal[self.cut + (i+1) * self.shift :
                                               self.duration + self.cut +
                                               (i+1)*self.shift, 0]

        return input_signal, target_signal

    def validate(self, prediction, target):
        """
        :param prediction: full time-series response of model
        :param target: target array
        :return: Memory capacity
        """

        # Evaluate correlation coefficient for all lags
        delay = np.array([i * self.shift
                          for i in reversed(range(1, self.num_lags + 1))])
        detcoef = np.zeros(self.num_lags)
        for i in range(self.num_lags):
            cor_coef = np.corrcoef(prediction[:, i], target[:, i])[0, 1]
            if np.isnan(cor_coef):
                detcoef[i] = 0.0
            else:
                detcoef[i] = cor_coef**2

        return np.sum(detcoef) * self.shift, delay, detcoef


class BinaryMemoryCapacityTask(MemoryCapacityTask):
    """
    Implements the MC task, but with binary instead of graded outputs.
    """

    def __init__(self, seed=None, **kwargs):
        """
        :param seed: for the RNG, if None (default), then it uses numpy
            RandomState's default seed
        :param kwargs: MemoryCapacityTask arguments
        """
        super().__init__(**kwargs)

        self._seed = seed
        if seed is None:
            self._rng = np.random.RandomState()
        else:
            self._rng = np.random.RandomState(self._seed)

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        self._rng.seed(self._seed)

    def generate_input_signal(self):
        """
        :return: the input signal of the task
        """

        return self._rng.binomial(n=1, p=0.5, size=(self.cut + self.duration
                                  + self.max_lag, 1)).astype(dtype=self.dtype)
