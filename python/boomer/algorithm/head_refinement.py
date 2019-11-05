#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for finding the optimal heads for rules that cover a certain region of the instance space.
"""

from abc import abstractmethod

import numpy as np
from boomer.algorithm._losses import Loss, DecomposableLoss
from boomer.algorithm._model import Head, FullHead, PartialHead, DTYPE_INDICES

from boomer.learners import Randomized


class HeadRefinement(Randomized):
    """
    A base class for all classes that allow to find heads that minimize a certain loss function.
    """

    def __init__(self, loss: Loss):
        """
        :param loss: The (surrogate) loss to be minimized
        """
        self.loss = loss

    @abstractmethod
    def find_head(self, expected_scores: np.ndarray, predicted_scores: np.ndarray) -> (Head, float):
        """
        Finds and returns the head that minimizes the loss function given expected and predicted scores.

        :param expected_scores:     An array of dtype float, shape `(num_examples, num_labels)`, representing the
                                    expected confidence scores according to the ground truth
        :param predicted_scores:    An array of dtype float, shape `(num_examples, num_labels`, representing the
                                    currently predicted confidence scores
        :return:                    The head, as well as its heuristic value
        """
        pass

    def find_default_head(self, expected_scores: np.ndarray, predicted_scores: np.ndarray) -> Head:
        """
        Finds and returns the head of the default rule that minimizes the loss function given expected and predicted
        scores.

        :param expected_scores:     An array of dtype float, shape `(num_examples, num_labels)`, representing the
                                    expected confidence scores according to the ground truth
        :param predicted_scores:    An array of dtype float, shape `(num_examples, num_labels`, representing the
                                    currently predicted confidence scores
        :return:                    The full head of the default rule
        """
        if isinstance(self.loss, DecomposableLoss):
            return HeadRefinement.__find_default_head_minimizing_decomposable_loss(expected_scores, predicted_scores,
                                                                                   self.loss)
        else:
            raise NotImplementedError("Non-decomposable loss functions not supported yet")

    @staticmethod
    def __find_default_head_minimizing_decomposable_loss(expected_scores: np.ndarray, predicted_scores: np.ndarray,
                                                         loss: DecomposableLoss) -> Head:
        """
        Finds and returns the head of the default rule that minimizes a decomposable loss function given expected and
        predicted scores.

        :param expected_scores:     An array of dtype float, shape `(num_examples, num_labels)`, representing the
                                    expected confidence scores according to the ground truth
        :param predicted_scores:    An array of dtype float, shape `(num_examples, num_labels`, representing the
                                    currently predicted confidence scores
        :param loss:                The (decomposable) loss function
        :return:
        """
        gradients = loss.calculate_gradients(expected_scores, predicted_scores)
        scores = loss.calculate_optimal_scores(gradients)
        return FullHead(scores)


class SingleLabelHeadRefinement(HeadRefinement):
    """
    Allows to find single-label heads that minimize a certain loss function.
    """

    def find_head(self, expected_scores: np.ndarray, predicted_scores: np.ndarray) -> (Head, float):
        if isinstance(self.loss, DecomposableLoss):
            return SingleLabelHeadRefinement.__find_head_minimizing_decomposable_loss(expected_scores, predicted_scores,
                                                                                      self.loss)
        else:
            raise NotImplementedError("Non-decomposable loss functions not supported yet")

    @staticmethod
    def __find_head_minimizing_decomposable_loss(expected_scores: np.ndarray, predicted_scores: np.ndarray,
                                                 loss: DecomposableLoss) -> (Head, float):
        """
        Finds and returns the head that minimizes a decomposable loss function given expected and predicted scores.

        :param expected_scores:     An array of dtype float, shape `(num_examples, num_labels)`, representing the
                                    expected confidence scores according to the ground truth
        :param predicted_scores:    An array of dtype float, shape `(num_examples, num_labels`, representing the
                                    currently predicted confidence scores
        :param loss:                The (decomposable) loss function
        :return:                    The head, as well as its heuristic value
        """
        gradients = loss.calculate_gradients(expected_scores, predicted_scores)
        scores = loss.calculate_optimal_scores(gradients)
        label_indices = np.linspace(0, scores.size, num=scores.size, endpoint=False, dtype=DTYPE_INDICES)
        h = loss.evaluate_predictions(scores, gradients)
        return PartialHead(label_indices, scores), h
