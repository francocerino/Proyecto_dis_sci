import numpy as np


def validate_parameters(input_value):
    if not isinstance(input_value, np.ndarray):
        raise TypeError(
            "The input parameters must be a numpy array."
            "Got instead type {}".format(type(input_value))
        )


def validate_training_set(input_value):
    if not isinstance(input_value, np.ndarray):
        raise TypeError(
            "The input training_set must be a numpy array."
            "Got instead type {}".format(type(input_value))
        )


def validate_physical_points(input_value):
    if not isinstance(input_value, np.ndarray):
        raise TypeError(
            "The input physical_points must be a numpy array."
            "Got instead type {}".format(type(input_value))
        )
