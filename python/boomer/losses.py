#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Implements additional loss functions for evaluating a model.
"""
import numpy as np
from scipy.sparse import issparse
from sklearn.utils.validation import check_consistent_length


def squared_error_loss(ground_truth, predictions) -> float:
    """
    Evaluates predictions according to the squared error loss.

    :param ground_truth:    An array of dtype float, shape `(num_examples, num_labels)`, representing the true labels
    :param predictions:     An array of dtype float, shape `(num_examples, num_labels)`, representing the predicted
                            numerical scores
    :return:                A scalar of dtype float, representing the evaluation score
    """
    if issparse(ground_truth):
        ground_truth = ground_truth.toarray()

    if issparse(predictions):
        predictions = predictions.toarray()

    check_consistent_length(ground_truth, predictions)
    return np.mean(np.square(ground_truth - predictions)).item()
