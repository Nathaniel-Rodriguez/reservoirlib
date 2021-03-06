from reservoirlib.utility import DEFAULT_FLOAT
from abc import ABC, abstractmethod
import numpy as np


class BaseMetric(ABC):
    """
    An abstract base that defines the interface for metrics
    """

    def __init__(self, dtype=None):
        """
        Initializes the Metric
        :param dtype: if necessary
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

    def __init__(self, cut=0, time_average=False, fractional_activity=False,
                 absolute_activity=False, **kwargs):
        """
        :param cut: The number of time-steps to drop
        :param time_average: Whether to divide activity by the number of
            times-steps. Default: False
        :param fractional_activity: whether to divide activity by the number of
            nodes in the network. Default: False
        :param absolute_activity: whether to take the absolute value of activity
            before sum. Default: False
        :param kwargs: BaseMetric arguments
        """

        super().__init__(**kwargs)
        self.cut = cut
        self.time_average = time_average
        self.fractional_activity = fractional_activity
        self.absolute_activity = absolute_activity

    def __call__(self, history):
        """
        :param history: TxN numpy array
        :return: size N numpy array
        """

        activity = np.sum(history[self.cut:], axis=1).copy()

        if self.absolute_activity:
            np.absolute(activity, out=activity)

        if self.time_average:
            remaining = history.shape[0] - self.cut
            np.divide(activity, remaining, out=activity)

        if self.fractional_activity:
            np.divide(activity, history.shape[1], out=activity)

        return activity


class ConvergenceMetric(BaseMetric):
    """
    Estimates whether an ESN has converged
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, history):
        """

        :param history: TxN numpy array
        :return: 1 (converged) or 0 (not converged)
        """

        if np.all(history[-2, :] == history[-1, :]):
            return 1
        else:
            return 0
