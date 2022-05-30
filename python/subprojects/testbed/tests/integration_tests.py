"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
import shutil
import subprocess
from abc import ABC
from functools import reduce
from os import path, makedirs
from typing import List, Optional
from unittest import TestCase

DIR_RES = 'python/subprojects/testbed/tests/res'

DIR_DATA = path.join(DIR_RES, 'data')

DIR_IN = path.join(DIR_RES, 'in')

DIR_OUT = path.join(DIR_RES, 'out')

DIR_RESULTS = path.join(path.join(DIR_RES, 'tmp'), 'results')

DIR_MODELS = path.join(path.join(DIR_RES, 'tmp'), 'models')

DATASET_EMOTIONS = 'emotions'

DATASET_ENRON = 'enron'

DATASET_LANGLOG = 'langlog'

DATASET_WEATHER = 'weather'

PRUNING_NO = 'none'

PRUNING_IREP = 'irep'

INSTANCE_SAMPLING_NO = 'none'

INSTANCE_SAMPLING_WITH_REPLACEMENT = 'with-replacement'

INSTANCE_SAMPLING_WITHOUT_REPLACEMENT = 'without-replacement'

INSTANCE_SAMPLING_STRATIFIED_LABEL_WISE = 'stratified-label-wise'

INSTANCE_SAMPLING_STRATIFIED_EXAMPLE_WISE = 'stratified-example-wise'

FEATURE_SAMPLING_NO = 'none'

FEATURE_SAMPLING_WITHOUT_REPLACEMENT = 'without-replacement'


class CmdBuilder:
    """
    A builder that allows to configure a command for running a rule learner.
    """

    def __init__(self, cmd: str, data_dir: str = DIR_DATA, dataset: str = DATASET_EMOTIONS):
        """
        :param cmd:         The command to be run
        :param data_dir:    The path of the directory that stores the dataset files
        :param dataset:     The name of the dataset
        """
        self.cmd = cmd
        self.output_dir = None
        self.model_dir = None
        self.folds = 0
        self.current_fold = 0
        self.training_data_evaluated = False
        self.evaluation_stored = True
        self.parameters_stored = False
        self.predictions_stored = False
        self.prediction_characteristics_stored = False
        self.data_characteristics_stored = False
        self.model_characteristics_stored = False
        self.rules_stored = False
        self.args = [cmd, '--log-level', 'DEBUG', '--data-dir', data_dir, '--dataset', dataset]
        self.tmp_dirs = []

    def set_output_dir(self, output_dir: Optional[str] = DIR_RESULTS):
        """
        Configures the rule learner to store output files in a given directory.

        :param output_dir:  The path of the directory where output files should be stored
        :return:            The builder itself
        """
        self.output_dir = output_dir

        if output_dir is not None:
            self.args.append('--output-dir')
            self.args.append(output_dir)
            self.tmp_dirs.append(output_dir)
        return self

    def set_model_dir(self, model_dir: Optional[str] = DIR_MODELS):
        """
        Configures the rule learner to store models in a given directory or load them, if available.

        :param model_dir:   The path of the directory where models should be stored
        :return:            The builder itself
        """
        self.model_dir = model_dir

        if model_dir is not None:
            self.args.append('--model-dir')
            self.args.append(model_dir)
            self.tmp_dirs.append(model_dir)
        return self

    def set_parameter_dir(self, parameter_dir: Optional[str] = DIR_IN):
        """
        Configures the rule learner to load parameter settings from a given directory, if available.

        :param parameter_dir:   The path of the directory, where parameter settings are stored
        :return:                The builder itself
        """
        if parameter_dir is not None:
            self.args.append('--parameter-dir')
            self.args.append(parameter_dir)
        return self

    def cross_validation(self, folds: int = 10, current_fold: int = 0):
        """
        Configures the rule learner to use a cross validation.

        :param folds:           The total number of folds
        :param current_fold:    The fold to be run or 0, if all folds should be run
        :return:                The builder itself
        """
        self.folds = folds
        self.current_fold = current_fold
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
        self.training_data_evaluated = evaluate_training_data
        self.args.append('--evaluate-training-data')
        self.args.append(str(evaluate_training_data).lower())
        return self

    def print_evaluation(self, print_evaluation: bool = True):
        """
        Configures whether the evaluation results should be printed on the console or not.

        :param print_evaluation:    True, if the evaluation results should be printed, False otherwise
        :return:                    The builder self
        """
        self.args.append('--print-evaluation')
        self.args.append(str(print_evaluation).lower())
        return self

    def store_evaluation(self, store_evaluation: bool = True):
        """
        Configures whether the evaluation results should be written into output files or not.

        :param store_evaluation:    True, if the evaluation results should be written into output files or not
        :return:                    The builder itself
        """
        self.evaluation_stored = store_evaluation
        self.args.append('--store-evaluation')
        self.args.append(str(store_evaluation).lower())
        return self

    def print_parameters(self, print_parameters: bool = True):
        """
        Configures whether the parameters should be printed on the console or not.

        :param print_parameters:    True, if the parameters should be printed, False otherwise
        :return:                    The builder itself
        """
        self.args.append('--print-parameters')
        self.args.append(str(print_parameters).lower())
        return self

    def store_parameters(self, store_parameters: bool = True):
        """
        Configures whether the parameters should be written into output files or not.

        :param store_parameters:    True, if the parameters should be written into output files, False otherwise
        :return:                    The builder itself
        """
        self.parameters_stored = store_parameters
        self.args.append('--store-parameters')
        self.args.append(str(store_parameters).lower())
        return self

    def print_predictions(self, print_predictions: bool = True):
        """
        Configures whether the predictions should be printed on the console or not.

        :param print_predictions:   True, if the predictions should be printed, False otherwise
        :return:                    The builder itself
        """
        self.args.append('--print-predictions')
        self.args.append(str(print_predictions).lower())
        return self

    def store_predictions(self, store_predictions: bool = True):
        """
        Configures whether the predictions should be written into output files or not.

        :param store_predictions:   True, if the predictions should be written into output files, False otherwise
        :return:                    The builder itself
        """
        self.predictions_stored = store_predictions
        self.args.append('--store-predictions')
        self.args.append(str(store_predictions).lower())
        return self

    def print_prediction_characteristics(self, print_prediction_characteristics: bool = True):
        """
        Configures whether the characteristics of predictions should be printed on the console or not.

        :param print_prediction_characteristics:    True, if the characteristics of predictions should be printed, False
                                                    otherwise
        :return:                                    The builder itself
        """
        self.args.append('--print-prediction-characteristics')
        self.args.append(str(print_prediction_characteristics).lower())
        return self

    def store_prediction_characteristics(self, store_prediction_characteristics: bool = True):
        """
        Configures whether the characteristics of predictions should be written into output files or not.

        :param store_prediction_characteristics:    True, if the characteristics of predictions should be written into
                                                    output files, False otherwise
        :return:                                    The builder itself
        """
        self.prediction_characteristics_stored = store_prediction_characteristics
        self.args.append('--store-prediction-characteristics')
        self.args.append(str(store_prediction_characteristics).lower())
        return self

    def print_data_characteristics(self, print_data_characteristics: bool = True):
        """
        Configures whether the characteristics of datasets should be printed on the console or not.

        :param print_data_characteristics:  True, if the characteristics of datasets should be printed, False otherwise
        :return:                            The builder itself
        """
        self.args.append('--print-data-characteristics')
        self.args.append(str(print_data_characteristics).lower())
        return self

    def store_data_characteristics(self, store_data_characteristics: bool = True):
        """
        Configures whether the characteristics of datasets should be written into output files or not.

        :param store_data_characteristics:  True, if the characteristics of datasets should be written into output
                                            files, False otherwise
        :return:                            The builder itself
        """
        self.data_characteristics_stored = store_data_characteristics
        self.args.append('--store-data-characteristics')
        self.args.append(str(store_data_characteristics).lower())
        return self

    def print_model_characteristics(self, print_model_characteristics: bool = True):
        """
        Configures whether the characteristics of models should be printed on the console or not.

        :param print_model_characteristics: True, if the characteristics of models should be printed, False otherwise
        :return:                            The builder itself
        """
        self.args.append('--print-model-characteristics')
        self.args.append(str(print_model_characteristics).lower())
        return self

    def store_model_characteristics(self, store_model_characteristics: bool = True):
        """
        Configures whether the characteristics of models should be written into output files or not.

        :param store_model_characteristics: True, if the characteristics of models should be written into output files,
                                            False otherwise
        :return:                            The builder itself
        """
        self.model_characteristics_stored = store_model_characteristics
        self.args.append('--store-model-characteristics')
        self.args.append(str(store_model_characteristics).lower())
        return self

    def print_rules(self, print_rules: bool = True):
        """
        Configures whether textual representations of the rules in a model should be printed on the console or not.

        :param print_rules: True, if textual representations of rules should be printed, False otherwise
        :return:            The builder itself
        """
        self.args.append('--print-rules')
        self.args.append(str(print_rules).lower())
        return self

    def store_rules(self, store_rules: bool = True):
        """
        Configures whether textual representations of the rules in a model should be written into output files or not.

        :param store_rules: True, if textual representations of rules should be written into output files, False
                            otherwise
        :return:            The builder itself
        """
        self.rules_stored = store_rules
        self.args.append('--store-rules')
        self.args.append(str(store_rules).lower())
        return self

    def one_hot_encoding(self, one_hot_encoding: bool = True):
        """
        Configures whether one-hot-encoding should be used to encode nominal feature values or not.

        :param one_hot_encoding:    True, if one-hot-encoding should be used, False otherwise
        :return:                    The builder itself
        """
        self.args.append('--one-hot-encoding')
        self.args.append(str(one_hot_encoding).lower())
        return self

    def sparse_feature_format(self, sparse: bool = True):
        """
        Configures whether sparse data structures should be used to represent the feature values of training examples or
        not.

        :param sparse:  True, if sparse data structures should be used to represent the feature values of training
                        examples, False otherwise
        :return:        The builder itself
        """
        self.args.append('--feature-format')
        self.args.append('sparse' if sparse else 'dense')
        return self

    def sparse_label_format(self, sparse: bool = True):
        """
        Configures whether sparse data structures should be used to represent the labels of training examples or not.

        :param sparse:  True, if sparse data structures should be used to represent the labels of training examples,
                        False otherwise
        :return:        The builder itself
        """
        self.args.append('--label-format')
        self.args.append('sparse' if sparse else 'dense')
        return self

    def sparse_predicted_label_format(self, sparse: bool = True):
        """
        Configures whether sparse data structures should be used to represent predicted labels or not.

        :param sparse:  True, if sparse data structures should be used to represent predicted labels, False otherwise
        :return:        The builder itself
        """
        self.args.append('--predicted-label-format')
        self.args.append('sparse' if sparse else 'dense')
        return self

    def instance_sampling(self, instance_sampling: str = INSTANCE_SAMPLING_WITHOUT_REPLACEMENT):
        """
        Configures the rule learner to sample from the available training examples.

        :param instance_sampling:   The name of the sampling method that should be used
        :return:                    The builder itself
        """
        self.args.append('--instance-sampling')
        self.args.append(instance_sampling)
        return self

    def feature_sampling(self, feature_sampling: str = FEATURE_SAMPLING_WITHOUT_REPLACEMENT):
        """
        Configures the rule learner to sample from the available features.

        :param feature_sampling:    The name of the sampling method that should be used
        :return:                    The builder itself
        """
        self.args.append('--feature-sampling')
        self.args.append(feature_sampling)
        return self

    def pruning(self, pruning: str = PRUNING_IREP):
        """
        Configures the rule learner to use a specific pruning method.

        :param pruning: The name of the pruning method that should be used
        :return:        The builder itself
        """
        self.args.append('--pruning')
        self.args.append(pruning)
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

    def __init__(self, methodName='runTest'):
        """
        :param methodName: The name of the test method to be executed
        """
        super(IntegrationTests, self).__init__(methodName)

    @staticmethod
    def __get_file_name(name: str, suffix: str, fold: Optional[int] = None):
        """
        Returns the name of an output file.

        :param name:    The name of the file
        :param suffix:  The suffix of the file
        :param fold:    The fold, the file corresponds to or None, if it does not correspond to a specific fold
        :return:        The name of the output file
        """
        if fold is not None:
            return name + '_fold-' + str(fold) + '.' + suffix
        else:
            return name + '_overall.' + suffix

    def __assert_file_exists(self, directory: str, file_name: str):
        """
        Asserts that a specific file exists.

        :param directory:   The path of the directory where the file should be located
        :param file_name:   The name of the file
        """
        file = path.join(directory, file_name)
        self.assertTrue(path.isfile(file), 'File ' + str(file) + ' does not exist')

    def __assert_files_exist(self, builder: CmdBuilder, directory: str, file_name: str, suffix: str):
        """
        Asserts that the files, which should be created by a command, exist.

        :param directory:   The directory where the files should be located
        :param file_name:   The name of the files
        :param suffix:      The suffix of the files
        :param builder:     The builder
        """
        if directory is not None:
            if builder.folds > 0:
                current_fold = builder.current_fold

                if current_fold > 0:
                    self.__assert_file_exists(directory, self.__get_file_name(file_name, suffix, current_fold))
                else:
                    for i in range(builder.folds):
                        self.__assert_file_exists(directory, self.__get_file_name(file_name, suffix, i + 1))
            else:
                self.__assert_file_exists(directory, self.__get_file_name(file_name, suffix))

    def __assert_model_files_exist(self, builder: CmdBuilder):
        """
        Asserts that the model files, which should be created by a command, exist.

        :param builder: The builder
        """
        self.__assert_files_exist(builder, builder.model_dir, builder.cmd, 'model')

    def __assert_output_files_exist(self, builder: CmdBuilder, file_name: str, suffix: str):
        """
        Asserts that output files, which should be created by a command, exist.

        :param builder: The builder
        """
        self.__assert_files_exist(builder, builder.output_dir, file_name, suffix)

    @staticmethod
    def __get_output_name(prefix: str, training_data: bool = False):
        """
        Returns the name of an output file (without suffix).

        :param prefix:          The prefix of the file name
        :param training_data:   True, if the output file corresponds to the training data, False otherwise
        :return:                The name of the output file
        """
        return prefix + '_' + ('training' if training_data else 'test')

    def __assert_evaluation_files_exist(self, builder: CmdBuilder):
        """
        Asserts that the evaluation files, which should be created by a command, exist.

        :param builder: The builder
        """
        if builder.evaluation_stored:
            prefix = 'evaluation'
            suffix = 'csv'
            self.__assert_output_files_exist(builder, self.__get_output_name(prefix), suffix)

            if builder.training_data_evaluated:
                self.__assert_output_files_exist(builder, self.__get_output_name(prefix, True), suffix)

    def __assert_parameter_files_exist(self, builder: CmdBuilder):
        """
        Asserts that the parameter files, which should be created by a command, exist.

        :param builder: The builder
        """
        if builder.parameters_stored:
            self.__assert_output_files_exist(builder, 'parameters', 'csv')

    def __assert_prediction_files_exist(self, builder: CmdBuilder):
        """
        Asserts that the prediction files, which should be created by a command, exist.

        :param builder: The builder
        """
        if builder.predictions_stored:
            prefix = 'predictions'
            suffix = 'arff'
            self.__assert_output_files_exist(builder, self.__get_output_name(prefix), suffix)

            if builder.training_data_evaluated:
                self.__assert_output_files_exist(builder, self.__get_output_name(prefix, True), suffix)

    def __assert_prediction_characteristic_files_exist(self, builder: CmdBuilder):
        """
        Asserts that the prediction characteristic files, which should be created by a command, exist.

        :param builder: The builder
        """
        if builder.prediction_characteristics_stored:
            prefix = 'prediction_characteristics'
            suffix = 'csv'
            self.__assert_output_files_exist(builder, self.__get_output_name(prefix), suffix)

            if builder.training_data_evaluated:
                self.__assert_output_files_exist(builder, self.__get_output_name(prefix, True), suffix)

    def __assert_data_characteristic_files_exist(self, builder: CmdBuilder):
        """
        Asserts that the data characteristic files, which should be created by a command, exist.

        :param builder: The builder
        """
        if builder.data_characteristics_stored:
            self.__assert_output_files_exist(builder, 'data_characteristics', 'csv')

    def __assert_model_characteristic_files_exist(self, builder: CmdBuilder):
        """
        Asserts that the model characteristic files, which should be created by a command, exist.

        :param builder: The builder
        """
        if builder.model_characteristics_stored:
            self.__assert_output_files_exist(builder, 'model_characteristics', 'csv')

    def __assert_rule_files_exist(self, builder: CmdBuilder):
        """
        Assert that the rule files, which shouldbe created by a command, exist.

        :param builder: The builder
        """
        if builder.rules_stored:
            self.__assert_output_files_exist(builder, 'rules', 'txt')

    @staticmethod
    def __remove_tmp_dirs(builder: CmdBuilder):
        """
        Removes the temporary directories that have been used by a command.

        :param builder: The builder
        """
        for tmp_dir in builder.tmp_dirs:
            shutil.rmtree(tmp_dir)

    @staticmethod
    def __format_cmd(args: List[str]):
        return reduce(lambda txt, arg: txt + (' ' + arg if len(txt) > 0 else arg), args, '')

    def __run_cmd(self, args: List[str]):
        """
        Runs a given command.

        :param args:    A list that stores the command, as well as its arguments
        :return:        The output of the command
        """
        out = subprocess.run(args, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0,
                         'Command "' + self.__format_cmd(args) + '" terminated with non-zero exit code\n\n' + str(
                             out.stderr))
        return out

    def run_cmd(self, builder: CmdBuilder, expected_output_file_name: str = None, expected_output_dir: str = DIR_OUT):
        """
        Runs a command that has been configured via a builder.

        :param builder:                     The builder
        :param expected_output_file_name:   The name of the text file that contains the expected output of the command
        :param expected_output_dir:         The path of the directory that contains the file with the expected output
        """
        for tmp_dir in builder.tmp_dirs:
            makedirs(tmp_dir, exist_ok=True)

        args = builder.build()
        out = self.__run_cmd(args)

        if builder.model_dir is not None:
            out = self.__run_cmd(args)

        if expected_output_file_name is not None:
            stdout = str(out.stdout).splitlines()

            with open(path.join(expected_output_dir, expected_output_file_name + '.txt'), 'r') as f:
                for i, line in enumerate(f):
                    line = line.strip('\n')

                    if not line.startswith('INFO Configuration:') and not line.endswith('seconds'):
                        self.assertEqual(stdout[i], line, 'Output differs at line ' + str(i + 1))

        self.__assert_model_files_exist(builder)
        self.__assert_evaluation_files_exist(builder)
        self.__assert_parameter_files_exist(builder)
        self.__assert_prediction_files_exist(builder)
        self.__assert_prediction_characteristic_files_exist(builder)
        self.__assert_data_characteristic_files_exist(builder)
        self.__assert_model_characteristic_files_exist(builder)
        self.__assert_rule_files_exist(builder)
        self.__remove_tmp_dirs(builder)
