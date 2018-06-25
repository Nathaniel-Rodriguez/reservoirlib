from reservoirlib.utility import DEFAULT_FLOAT
from abc import ABC, abstractmethod
import numpy as np


class BaseMetric(ABC):
    """
    An abstract base that defines the interface for metrics
    """

    def __init__(self, dtype=None):
        """
        Initializes the trainer
        :param dtype: the numpy type of the output solutions
        """

        if dtype is None:
            self.dtype = DEFAULT_FLOAT
        else:
            self.dtype = dtype

    @abstractmethod
    def __call__(self, history):
        """
        Uses the history of the model to calculate some metric
        :param history: numpy array with shape TxN
            T=number of time-steps
            N=number of neurons
        :return: some value
        """
        pass


class ActivationMetric(BaseMetric):
    """
    Calculates the level of activation of neurons in a neural network.
    """

    def __init__(self, cut=0, time_average=False, **kwargs):
        """
        :param cut: The number of time-steps to drop
        :param time_average: Whether to divide activity by the number of
            times-steps. Default: False
        :param kwargs: BaseMetric arguments
        """

        super().__init__(**kwargs)
        self.cut = cut
        self.time_average = time_average

    def __call__(self, history):
        """
        :param history: TxN numpy array
        :return: size N numpy array
        """

        activity = np.sum(history[self.cut:], axis=1)
        if self.time_average:
            remaining = history.shape[0] - self.cut
            np.divide(activity, remaining, out=activity)

        return activity
