import numpy as np
from functools import partial


class Distribution(np.random.RandomState):
    """
    A sub-class around random state that is specialized to a specific
    distribution. The user can pick from the distribution methods available
    to RandomState and provide arguments to it while leaving out any they
    wish to specify at call time.

    When called, the Distribution object will generate random values as a numpy
    array.

    example:
        Distribution("multivariate_normal",
                     {'mean': np.zeros(num_variables),
                     'cov': covariance_matrix / fitness},
                     seed=seed)
    """

    def __init__(self, distribution, distribution_args=None, *args, **kwargs):
        """
        :param distribution: string of one of the methods available to RandomState
        :param distribution_args: dictionary of argument/value pairs. Default: None
        :param args: any other RandomState initialization arguments
        :param kwargs: any other RandomState keyword arguments
        """
        if distribution_args is None:
            distribution_args = {}

        super().__init__(*args, **kwargs)
        self._partial_distribution_method = partial(getattr(self, distribution),
                                                    **distribution_args)

    def __call__(self, *args, **kwargs):
        """
        :param kwargs: Any positional or keyword arguments NOT provided at
            initialization.
        :return: numpy array of random values generated from the distribution.
        """
        return self._partial_distribution_method(*args, **kwargs)

    def set_distribution(self, distribution, distribution_args=None):
        """
        Changes the call functionality by using a different distribution
        :param distribution: string of one of the methods available to RandomState
        :param distribution_args: dictionary of argument/value pairs. Default: None
        """
        if distribution_args is None:
            distribution_args = {}

        self._partial_distribution_method = partial(getattr(self, distribution),
                                                    **distribution_args)
