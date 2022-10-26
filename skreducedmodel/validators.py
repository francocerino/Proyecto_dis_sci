import numpy as np


def validate_parameters(input):
    if not isinstance(input, np.ndarray):
        raise TypeError(
            "The input parameters must be a numpy array."
            "Got instead type {}".format(type(input))
        )


def validate_training_set(input):
    if not isinstance(input, np.ndarray):
        raise TypeError(
            "The input training_set must be a numpy array."
            "Got instead type {}".format(type(input))
        )


def validate_physical_points(input):
    if not isinstance(input, np.ndarray):
        raise TypeError(
            "The input physical_points must be a numpy array."
            "Got instead type {}".format(type(input))
        )
