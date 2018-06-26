import numpy as np
import pickle


DEFAULT_INT = np.int64
DEFAULT_FLOAT = np.float64


def load(filename):
    pickled_obj_file = open(filename, 'rb')
    obj = pickle.load(pickled_obj_file)
    pickled_obj_file.close()

    return obj


def save(obj, filename):
    """
    objectives and their args are not saved with the ES
    """
    pickled_obj_file = open(filename, 'wb')
    pickle.dump(obj, pickled_obj_file, protocol=pickle.DEFAULT_PROTOCOL)
    pickled_obj_file.close()


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
