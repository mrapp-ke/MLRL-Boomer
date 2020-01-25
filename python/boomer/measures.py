#!/usr/bin/python


"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides measures for performance evaluation.
"""
import numpy as np


def squared_error_loss(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculates and return an evaluation score according to the (macro-averaged) squared error loss.

    :param ground_truth:    An array of dtype float, shape `(num_examples, num_labels)`, representing the ground truth
    :param predictions:     An array of dtype float, shape `(num_examples, num_labels)`, representing the predictions
    :return:                A scalar of dtype float, representing the calculated evaluation score
    """
    ground_truth = np.where(ground_truth > 0, 1, -1)
    predictions = np.clip(predictions, -1, 1)
    return np.average(np.square(ground_truth - predictions)).item()


def logistic_loss(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculates and returns an evaluation score according to the (example-based) logistic loss.

    :param ground_truth:    An array of dtype float, shape `(num_examples, num_labels)`, representing the ground truth
    :param predictions:     An array of dtype float, shape `(num_examples, num_labels)`, representing the predictions
    :return:                A scalar of dtype float, representing the calculated evaluation score
    """
    ground_truth = np.where(ground_truth > 0, 1, -1)
    predictions = np.clip(predictions, -1, 1)
    return np.average(np.log(1 + np.sum(np.exp(np.multiply(-ground_truth, predictions)), axis=1)))
