#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides functions for refining and evaluating candidate rules.
"""
import numpy as np

from boomer.algorithm.losses import Loss, DecomposableLoss
from boomer.algorithm.model import Rule, ConjunctiveBody, Head
from boomer.algorithm.stats import get_num_features, get_num_examples


class AllFeaturesIterator:

    def __init__(self, x: np.ndarray):
        self.num_features = get_num_features(x)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < self.num_features - 1:
            result = self.i
            self.i += 1
            return result
        else:
            raise StopIteration


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


def refine_rule(x: np.ndarray, x_sorted_indices: np.ndarray, expected_scores: np.ndarray, predicted_scores: np.ndarray,
                loss: Loss, feature_iterator) -> Rule:
    current_h = None
    leq_conditions = {}  # TODO Specify type
    gr_conditions = {}  # TODO Specify type
    head = None

    while True:
        refinement = __get_best_refinement(x, x_sorted_indices, expected_scores, predicted_scores, loss,
                                           feature_iterator)

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
            x_sorted_indices = np.argsort(x, axis=0)
        else:
            break

    return __build_rule(leq_conditions, gr_conditions, head)


def __get_best_refinement(x: np.ndarray, x_sorted_indices: np.ndarray, expected_scores: np.ndarray,
                          predicted_scores: np.ndarray, loss: Loss, feature_iterator) -> Refinement:
    refinement = None

    for c in iter(feature_iterator):
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
    head, gradients = loss.derive_scores(expected_scores, predicted_scores)
    h = loss.evaluate_predictions(head, gradients)
    return head, h


def __build_rule(leq_conditions, gr_conditions, head: Head) -> Rule:
    leq_features = np.fromiter(leq_conditions.keys(), dtype=np.int32)
    leq_thresholds = np.fromiter(leq_conditions.values(), dtype=np.float32)
    gr_features = np.fromiter(gr_conditions.keys(), dtype=np.int32)
    gr_thresholds = np.fromiter(gr_conditions.values(), dtype=np.float32)
    body = ConjunctiveBody(leq_features, leq_thresholds, gr_features, gr_thresholds)
    return Rule(body, head)
