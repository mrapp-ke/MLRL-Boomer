"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides functions for parsing command line arguments that are specific to boosting algorithms.
"""
from argparse import ArgumentParser

PARAM_SHRINKAGE = '--shrinkage'

PARAM_L1_REGULARIZATION_WEIGHT = '--l1-regularization-weight'

PARAM_L2_REGULARIZATION_WEIGHT = '--l2-regularization-weight'

PARAM_STATISTIC_FORMAT = '--statistic-format'

PARAM_DEFAULT_RULE = '--default-rule'

PARAM_LABEL_BINNING = '--label-binning'

PARAM_LOSS = '--loss'

PARAM_MARGINAL_PROBABILITY_CALIBRATION = '--marginal-probability-calibration'

PARAM_JOINT_PROBABILITY_CALIBRATION = '--joint-probability-calibration'

PARAM_BINARY_PREDICTOR = '--binary-predictor'

PARAM_PROBABILITY_PREDICTOR = '--probability-predictor'


def add_shrinkage_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_SHRINKAGE,
                        type=float,
                        help='The shrinkage parameter, a.k.a. the learning rate, to be used. Must be in (0, 1].')


def add_regularization_arguments(parser: ArgumentParser):
    parser.add_argument(PARAM_L1_REGULARIZATION_WEIGHT,
                        type=float,
                        help='The weight of the L1 regularization. Must be at least 0.')
    parser.add_argument(PARAM_L2_REGULARIZATION_WEIGHT,
                        type=float,
                        help='The weight of the L2 regularization. Must be at least 0.')
