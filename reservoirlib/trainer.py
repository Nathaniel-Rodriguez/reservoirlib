from abc import ABC
from scipy import linalg
from reservoirlib.utilities import DEFAULT_FLOAT


class BaseTrainer(ABC):
    """
    An abstract base that defines the interface for trainers
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
    def __call__(self, stacked_history, stacked_target):
        """
        Uses the history of the model and the target to find out what linear
        set of parameters best matches target
        :param stacked_history: numpy array with shape Sx(N+K)
        :param stacked_target: numpy array with shape
        :return: the parameters that minimize the error
        """
        pass


class LeastSquaredErrorTrainer(BaseTrainer):
    """
    Trainer that minimizes the normalized root mean square error
    """

    def __init__(self, **kwargs):
        """
        :param kwargs: BaseTrainer arguments
        """
        super().__init__(**kwargs)

    def __call__(self, stacked_history, stacked_target):
        """
        :param stacked_history: numpy array with shape Sx(N+K)
        :param stacked_target: numpy array with shape SxO
        :return: (N+K)xO
        """

        # Run regression
        # input: Sx(N+K) and SxO
        # solution: (N+K)xO
        solution, residuals, rank, sing = linalg.lstsq(stacked_history,
                                                       stacked_target)
        return solution
