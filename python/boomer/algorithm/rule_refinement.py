#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides functions for refining and evaluating candidate rules.
"""
from typing import Dict

import numpy as np

from boomer.algorithm.losses import Loss, DecomposableLoss
from boomer.algorithm.model import Rule, ConjunctiveBody, Head, DTYPE_INDICES, DTYPE_FEATURES
from boomer.algorithm.stats import get_num_features, get_num_examples
from boomer.algorithm.sub_sampling import FeatureSubSampling


class Refinement:

    def __init__(self, h: float, leq: bool, threshold: float, feature_index: int, threshold_index: int, head: Head,
                 covered_indices: np.ndarray):
        self.h = h
        self.leq = leq
        self.threshold = threshold
        self.feature_index = feature_index
        self.threshold_index = threshold_index
        self.head = head
        self.covered_indices = covered_indices


def presort_features(x: np.ndarray) -> np.ndarray:
    """
    Column-wise sorts a given feature matrix.

    :param x:   An array of dtype float, shape `(num_examples, num_features)`, representing the features of the training
                examples
    :return:    An array of dtype int, shape `(num_examples, num_features)`, representing the row-indices of the
                original examples at a certain position when sorting column-wise
    """
    x_sorted_indices = np.argsort(x, axis=0)
    return x_sorted_indices if x_sorted_indices.flags.fortran else np.asfortranarray(x_sorted_indices,
                                                                                     dtype=DTYPE_INDICES)


def grow_rule(x: np.ndarray, expected_scores: np.ndarray, predicted_scores: np.ndarray, loss: Loss, random_state: int,
              feature_sub_sampling: FeatureSubSampling = None, presorted_indices: np.ndarray = None) -> Rule:
    current_h = None
    leq_conditions: Dict[int, float] = {}
    gr_conditions: Dict[int, float] = {}
    head = None
    i = 1

    while True:
        refinement = __find_best_refinement(x, expected_scores, predicted_scores, loss, feature_sub_sampling,
                                            presorted_indices, iteration=i, random_state=random_state)

        if refinement is not None and (current_h is None or refinement.h < current_h):
            current_h = refinement.h
            head = refinement.head

            if refinement.leq:
                leq_conditions[refinement.feature_index] = refinement.threshold
            else:
                gr_conditions[refinement.feature_index] = refinement.threshold

            x = x[refinement.covered_indices]
            expected_scores = expected_scores[refinement.covered_indices]
            predicted_scores = predicted_scores[refinement.covered_indices]
            i += 1
        else:
            break

    return __create_rule(leq_conditions, gr_conditions, head)


def __sub_sample_features(x: np.ndarray, presorted_indices: np.ndarray, feature_sub_sampling: FeatureSubSampling,
                          iteration: int, random_state: int) -> (np.ndarray, np.ndarray):
    if feature_sub_sampling is None:
        return x, presorted_indices
    else:
        feature_sub_sampling.random_state = iteration * random_state
        sample_indices = feature_sub_sampling.sub_sample(x)
        return x[sample_indices], presorted_indices[sample_indices]


def __find_best_refinement(x: np.ndarray, expected_scores: np.ndarray, predicted_scores: np.ndarray, loss: Loss,
                           feature_sub_sampling: FeatureSubSampling, presorted_indices: np.ndarray,
                           iteration: int, random_state: int) -> Refinement:
    x, presorted_indices = __sub_sample_features(x, presorted_indices, feature_sub_sampling, iteration=iteration,
                                                 random_state=random_state)
    x_sorted_indices = presorted_indices if presorted_indices is not None else presort_features(x)
    refinement = None

    for c in range(0, get_num_features(x)):
        indices = x_sorted_indices[:, c]
        expected_scores_sorted = expected_scores[indices]
        predicted_scores_sorted = predicted_scores[indices]

        for r in range(1, get_num_examples(x) - 1):
            current_threshold = x[indices[r - 1], c]
            next_threshold = x[indices[r], c]

            if current_threshold != next_threshold and not np.array_equal(expected_scores_sorted[r - 1, :],
                                                                          expected_scores_sorted[r, :]):
                # LEQ
                expected_scores_subset = expected_scores_sorted[:r, :]
                predicted_scores_subset = predicted_scores_sorted[:r, :]
                head, h = __derive_optimal_head(expected_scores_subset, predicted_scores_subset, loss)

                if refinement is None or h <= refinement.h:
                    if refinement is None:
                        refinement = Refinement(h=h, leq=True,
                                                threshold=__get_threshold(current_threshold, next_threshold),
                                                feature_index=c, threshold_index=r, head=head,
                                                covered_indices=indices[:r])
                    else:
                        refinement.h = h
                        refinement.leq = True
                        refinement.threshold = __get_threshold(current_threshold, next_threshold)
                        refinement.feature_index = c
                        refinement.threshold_index = r
                        refinement.head = head
                        refinement.covered_indices = indices[:r]

                # GR
                expected_scores_subset = expected_scores_sorted[r:, :]
                predicted_scores_subset = predicted_scores_sorted[r:, :]
                head, h = __derive_optimal_head(expected_scores_subset, predicted_scores_subset, loss)

                if h < refinement.h:
                    refinement.h = h
                    refinement.leq = False
                    refinement.threshold = __get_threshold(current_threshold, next_threshold)
                    refinement.threshold_index = r
                    refinement.head = head
                    refinement.covered_indices = indices[r:]

    return refinement


def __get_threshold(current_threshold: float, next_threshold: float) -> float:
    return current_threshold + ((next_threshold - current_threshold) / 2)


def __derive_optimal_head(expected_scores: np.ndarray, predicted_scores: np.ndarray, loss: Loss) -> (Head, float):
    if isinstance(loss, DecomposableLoss):
        return __derive_optimal_head_using_decomposable_loss(expected_scores, predicted_scores, loss)
    else:
        # TODO: Implement
        raise NotImplementedError('Non-decomposable loss functions not supported yet...')


def __derive_optimal_head_using_decomposable_loss(expected_scores: np.ndarray, predicted_scores: np.ndarray,
                                                  loss: DecomposableLoss) -> (Head, float):
    scores, gradients = loss.derive_scores(expected_scores, predicted_scores)
    h = loss.evaluate_predictions(scores, gradients)
    return Head(np.linspace(0, scores.size, num=scores.size, endpoint=False, dtype=DTYPE_INDICES), scores), h


def __create_rule(leq_conditions: Dict[int, float], gr_conditions: Dict[int, float], head: Head) -> Rule:
    """
    Creates and returns a new rule with certain conditions and a specific head.

    :param leq_conditions:  A 'Dict' that contains the conditions that use the "less-or-equal" operator. The keys
                            correspond to the feature indices and the values denote the thresholds to be used by the
                            conditions
    :param gr_conditions:   A 'Dict' that contains the conditions that use the "greater" operator. The keys correspond
                            to the feature indices and the values denote the thresholds to be used by the conditions
    :param head:            The 'Head' of the rule
    :return:                The 'Rule' that has been created
    """
    leq_features = np.fromiter(leq_conditions.keys(), dtype=DTYPE_INDICES)
    leq_thresholds = np.fromiter(leq_conditions.values(), dtype=DTYPE_FEATURES)
    gr_features = np.fromiter(gr_conditions.keys(), dtype=DTYPE_INDICES)
    gr_thresholds = np.fromiter(gr_conditions.values(), dtype=DTYPE_FEATURES)
    body = ConjunctiveBody(leq_features, leq_thresholds, gr_features, gr_thresholds)
    return Rule(body, head)
