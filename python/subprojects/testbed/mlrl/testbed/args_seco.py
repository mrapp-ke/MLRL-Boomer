"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides functions for parsing command line arguments that are specific to separate-and-conquer (SeCo) algorithms.
"""
from argparse import ArgumentParser

from mlrl.common.format import format_dict_keys, format_string_set
from mlrl.seco.config import LIFT_FUNCTION_VALUES, HEAD_TYPE_PARTIAL, HEAD_TYPE_VALUES
from mlrl.testbed.args import PARAM_HEAD_TYPE

PARAM_LIFT_FUNCTION = '--lift-function'

PARAM_HEURISTIC = '--heuristic'

PARAM_PRUNING_HEURISTIC = '--pruning-heuristic'


def add_head_type_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_HEAD_TYPE, type=str,
                        help='The type of the rule heads that should be used. Must be one of '
                             + format_string_set(HEAD_TYPE_VALUES) + '.')


def add_lift_function_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_LIFT_FUNCTION, type=str,
                        help='The lift function to be used for the induction of multi-label rules. Must be one of '
                             + format_dict_keys(LIFT_FUNCTION_VALUES) + '. Does only have an effect if the parameter '
                             + PARAM_HEAD_TYPE + ' is set to "' + HEAD_TYPE_PARTIAL + '".')
