"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
import subprocess
from abc import ABC
from os import path
from typing import List
from unittest import TestCase

CMD_BOOMER = 'boomer'

DIR_RES = 'python/subprojects/testbed/tests/res'

DIR_DATA = path.join(DIR_RES, 'data')

DIR_OUT = path.join(DIR_RES, 'out')

DATASET_EMOTIONS = 'emotions'


class CmdBuilder:
    """
    A builder that allows to configure the command for running a rule learner.
    """

    def __init__(self, cmd: str = CMD_BOOMER, data_dir: str = DIR_DATA, dataset: str = DATASET_EMOTIONS):
        """
        :param cmd:         The command to be run
        :param data_dir:    The path of the directory that stores the dataset files
        :param dataset:     The name of the dataset
        """
        self.args = [cmd, '--log-level', 'DEBUG', '--data-dir', data_dir, '--dataset', dataset]

    def cross_validation(self, folds: int = 10, current_fold: int = 0):
        """
        Configures the rule learner to use a cross validation.

        :param folds:           The total number of folds
        :param current_fold:    The fold to be run or 0, if all folds should be run
        :return:                The builder itself
        """
        self.args.append('--folds')
        self.args.append(str(folds))
        self.args.append('--current-fold')
        self.args.append(str(current_fold))
        return self

    def evaluate_training_data(self, evaluate_training_data: bool = True):
        """
        Configures whether the rule learner should be evaluated on the training data or not.

        :param evaluate_training_data:  True, if the rule learner should be evaluated on the training data, False
                                        otherwise
        :return:                        The builder itself
        """
        self.args.append('--evaluate-training-data')
        self.args.append(str(evaluate_training_data).lower())
        return self

    def build(self) -> List[str]:
        """
        Returns a list of strings that contains the command that has been configured using the builder, as well as all
        of its arguments.

        :return: The command that has been configured
        """
        return self.args


class IntegrationTests(ABC, TestCase):
    """
    An abstract base class for all integration tests.
    """

    def run_cmd(self, builder: CmdBuilder, expected_output_file_name: str = None, expected_output_dir: str = DIR_OUT):
        """
        Runs a command that has been configured via a builder.

        :param builder:                     The builder
        :param expected_output_file_name:   The name of the text file that contains the expected output of the command
        :param expected_output_dir:         The path of the directory that contains the file with the expected output
        """
        args = builder.build()
        out = subprocess.run(args, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, 'Command terminated with non-zero exit code')

        if expected_output_file_name is not None:
            stdout = str(out.stdout).splitlines()

            with open(path.join(expected_output_dir, expected_output_file_name + '.txt'), 'r') as f:
                for i, line in enumerate(f):
                    line = line.strip('\n')

                    if not line.startswith('INFO Configuration:') and not line.endswith('seconds'):
                        self.assertEqual(stdout[i], line, 'Output differs at line ' + str(i + 1))
