"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
import os
import re
import shutil
import subprocess

from abc import ABC, abstractmethod
from functools import reduce
from os import makedirs, path
from sys import platform
from typing import Any, List, Optional
from unittest import SkipTest, TestCase

from mlrl.testbed.io import ENCODING_UTF8

OVERWRITE_EXPECTED_OUTPUT_FILES = False

DIR_RES = 'python/subprojects/testbed/tests/res'

DIR_DATA = path.join(DIR_RES, 'data')

DIR_IN = path.join(DIR_RES, 'in')

DIR_OUT = path.join(DIR_RES, 'out')

DIR_RESULTS = path.join(path.join(DIR_RES, 'tmp'), 'results')

DIR_MODELS = path.join(path.join(DIR_RES, 'tmp'), 'models')

DATASET_EMOTIONS = 'emotions'

DATASET_EMOTIONS_NOMINAL = 'emotions-nominal'

DATASET_EMOTIONS_ORDINAL = 'emotions-ordinal'

DATASET_ENRON = 'enron'

DATASET_LANGLOG = 'langlog'

DATASET_BREAST_CANCER = 'breast-cancer'

DATASET_MEKA = 'meka'

DATASET_ATP7D = 'atp7d'

DATASET_ATP7D_NUMERICAL_SPARSE = 'atp7d-numerical-sparse'

DATASET_ATP7D_NOMINAL = 'atp7d-nominal'

DATASET_ATP7D_BINARY = 'atp7d-binary'

DATASET_ATP7D_ORDINAL = 'atp7d-ordinal'

DATASET_ATP7D_MEKA = 'atp7d-meka'

DATASET_HOUSING = 'housing'

RULE_PRUNING_NO = 'none'

RULE_PRUNING_IREP = 'irep'

RULE_INDUCTION_TOP_DOWN_GREEDY = 'top-down-greedy'

RULE_INDUCTION_TOP_DOWN_BEAM_SEARCH = 'top-down-beam-search'

INSTANCE_SAMPLING_NO = 'none'

INSTANCE_SAMPLING_WITH_REPLACEMENT = 'with-replacement'

INSTANCE_SAMPLING_WITHOUT_REPLACEMENT = 'without-replacement'

FEATURE_SAMPLING_NO = 'none'

FEATURE_SAMPLING_WITHOUT_REPLACEMENT = 'without-replacement'

OUTPUT_SAMPLING_NO = 'none'

OUTPUT_SAMPLING_WITHOUT_REPLACEMENT = 'without-replacement'

OUTPUT_SAMPLING_ROUND_ROBIN = 'round-robin'

HOLDOUT_NO = 'none'

HOLDOUT_RANDOM = 'random'

FEATURE_BINNING_EQUAL_WIDTH = 'equal-width'

FEATURE_BINNING_EQUAL_FREQUENCY = 'equal-frequency'


def skip_test_on_ci(decorated_function):
    """
    A decorator that disables all annotated test case if run on a continuous integration system.
    """

    def wrapper(*args, **kwargs):
        if os.getenv('GITHUB_ACTIONS') == 'true':
            raise SkipTest('Temporarily disabled when run on CI')

        decorated_function(*args, **kwargs)

    return wrapper


class CmdBuilder:
    """
    A builder that allows to configure a command for running a rule learner.
    """

    class AssertionCallback(ABC):
        """
        Must be implemented by classes that should be notified about test failures.
        """

        @abstractmethod
        def on_assertion_failure(self, message: str):
            """
            Must be implemented by subclasses in order to be notified about test failures.

            :param message: A message that indicates why a test has failed
            """

    def __init__(self,
                 callback: AssertionCallback,
                 expected_output_dir: str,
                 model_file_name: str,
                 runnable_module_name: str,
                 runnable_class_name: Optional[str] = None,
                 data_dir: str = DIR_DATA,
                 dataset: str = DATASET_EMOTIONS):
        """
        :param callback:                The callback that should be notified about test failures
        :param expected_output_dir:     The path of the directory that contains the file with the expected output
        :param model_file_name:         The name of files storing models that have been saved to disk (without suffix)
        :param runnable_module_name:    The fully qualified name of the runnable to be invoked by the 'testbed' program
        :param runnable_class_name:     The class name of the runnable to be invoked by the 'testbed' program
        :param data_dir:                The path of the directory that stores the dataset files
        :param dataset:                 The name of the dataset
        """
        self.callback = callback
        self.expected_output_dir = expected_output_dir
        self.model_file_name = model_file_name
        self.output_dir = None
        self.parameter_dir = None
        self.model_dir = None
        self.num_folds = 0
        self.current_fold = 0
        self.training_data_evaluated = False
        self.separate_train_test_sets = True
        self.evaluation_stored = True
        self.parameters_stored = False
        self.predictions_stored = False
        self.prediction_characteristics_stored = False
        self.data_characteristics_stored = False
        self.model_characteristics_stored = False
        self.rules_stored = False
        self.tmp_dirs = []
        self.args = self.__create_args(runnable_module_name=runnable_module_name,
                                       runnable_class_name=runnable_class_name,
                                       data_dir=data_dir,
                                       dataset=dataset)

    @staticmethod
    def __create_args(runnable_module_name: str, runnable_class_name: Optional[str], data_dir: str, dataset: str):
        args = ['testbed', runnable_module_name]

        if runnable_class_name:
            args.extend(['-r', runnable_class_name])

        args.extend(['--log-level', 'DEBUG', '--data-dir', data_dir, '--dataset', dataset])
        return args

    @staticmethod
    def __format_cmd(args: List[str]):
        return reduce(lambda txt, arg: txt + (' ' + arg if len(txt) > 0 else arg), args, '')

    def __run_cmd(self):
        """
        Runs the command that has been configured via the builder.

        :return: The output of the command
        """
        out = subprocess.run(self.args, capture_output=True, text=True, check=False)

        if out.returncode != 0:
            self.callback.on_assertion_failure('Command "' + self.__format_cmd(self.args)
                                               + '" terminated with non-zero exit code\n\n' + str(out.stderr))

        return out

    @staticmethod
    def __replace_durations_with_placeholders(line: str) -> str:
        regex_duration = '(\\d+ (day(s)*|hour(s)*|minute(s)*|second(s)*|millisecond(s)*))'
        return re.sub(regex_duration + '((, )' + regex_duration + ')*' + '(( and )' + regex_duration + ')?',
                      '<duration>', line)

    def __overwrite_output_file(self, stdout, expected_output_file):
        with open(expected_output_file, 'w', encoding=ENCODING_UTF8) as file:
            for line in stdout:
                line = self.__replace_durations_with_placeholders(line)
                line = line + '\n'
                file.write(line)

    def __assert_output_files_are_equal(self, stdout, expected_output_file):
        with open(expected_output_file, 'r', encoding=ENCODING_UTF8) as file:
            for i, expected_line in enumerate(file):
                expected_line = expected_line.strip('\n')
                line = stdout[i]
                line = line.strip('\n')
                line = self.__replace_durations_with_placeholders(line)

                if expected_line != line:
                    self.callback.on_assertion_failure('Output of command "' + self.__format_cmd(self.args)
                                                       + '" differs at line ' + str(i + 1) + ': Should be "'
                                                       + expected_line + '", but is "' + line + '"')

    def __remove_tmp_dirs(self):
        """
        Removes the temporary directories that have been used by a command.
        """
        for tmp_dir in self.tmp_dirs:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def __assert_model_files_exist(self):
        """
        Asserts that the model files, which should be created by a command, exist.
        """
        self._assert_files_exist(self.model_dir, self.model_file_name, 'model')

    def __assert_evaluation_files_exist(self):
        """
        Asserts that the evaluation files, which should be created by a command, exist.
        """
        if self.evaluation_stored:
            prefix = 'evaluation'
            suffix = 'csv'
            training_data = not self.separate_train_test_sets
            self._assert_output_files_exist(self._get_output_file_name(prefix, training_data=training_data), suffix)

            if self.training_data_evaluated:
                self._assert_output_files_exist(self._get_output_file_name(prefix, training_data=True), suffix)

    def __assert_parameter_files_exist(self):
        """
        Asserts that the parameter files, which should be created by a command, exist.
        """
        if self.parameters_stored:
            self._assert_files_exist(self.parameter_dir, 'parameters', 'csv')

    def __assert_prediction_files_exist(self):
        """
        Asserts that the prediction files, which should be created by a command, exist.
        """
        if self.predictions_stored:
            prefix = 'predictions'
            suffix = 'arff'
            training_data = not self.separate_train_test_sets
            self._assert_output_files_exist(self._get_output_file_name(prefix, training_data=training_data), suffix)

            if self.training_data_evaluated:
                self._assert_output_files_exist(self._get_output_file_name(prefix, training_data=True), suffix)

    def __assert_prediction_characteristic_files_exist(self):
        """
        Asserts that the prediction characteristic files, which should be created by a command, exist.
        """
        if self.prediction_characteristics_stored:
            prefix = 'prediction_characteristics'
            suffix = 'csv'
            training_data = not self.separate_train_test_sets
            self._assert_output_files_exist(self._get_output_file_name(prefix, training_data=training_data), suffix)

            if self.training_data_evaluated:
                self._assert_output_files_exist(self._get_output_file_name(prefix, training_data=True), suffix)

    def __assert_data_characteristic_files_exist(self):
        """
        Asserts that the data characteristic files, which should be created by a command, exist.
        """
        if self.data_characteristics_stored:
            self._assert_output_files_exist('data_characteristics', 'csv')

    def __assert_model_characteristic_files_exist(self):
        """
        Asserts that the model characteristic files, which should be created by a command, exist.
        """
        if self.model_characteristics_stored:
            self._assert_output_files_exist('model_characteristics', 'csv')

    def __assert_rule_files_exist(self):
        """
        Asserts that the rule files, which should be created by a command, exist.
        """
        if self.rules_stored:
            self._assert_output_files_exist('rules', 'txt')

    @staticmethod
    def _get_output_file_name(prefix: str, training_data: bool = False):
        """
        Returns the name of an output file (without suffix).

        :param prefix:          The prefix of the file name
        :param training_data:   True, if the output file corresponds to the training data, False otherwise
        :return:                The name of the output file
        """
        return prefix + '_' + ('training' if training_data else 'test')

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
        return name + '_overall.' + suffix

    def __assert_file_exists(self, directory: str, file_name: str):
        """
        Asserts that a specific file exists.

        :param directory:   The path of the directory where the file should be located
        :param file_name:   The name of the file
        """
        file = path.join(directory, file_name)

        if not path.isfile(file):
            self.callback.on_assertion_failure('Command "' + self.__format_cmd(self.args)
                                               + '" is expected to create file ' + str(file)
                                               + ', but it does not exist')

    def _assert_files_exist(self, directory: str, file_name: str, suffix: str):
        """
        Asserts that the files, which should be created by a command, exist.

        :param directory:   The directory where the files should be located
        :param file_name:   The name of the files
        :param suffix:      The suffix of the files
        """
        if directory is not None:
            if self.num_folds > 0:
                current_fold = self.current_fold

                if current_fold > 0:
                    self.__assert_file_exists(directory, self.__get_file_name(file_name, suffix, current_fold))
                else:
                    for i in range(self.num_folds):
                        self.__assert_file_exists(directory, self.__get_file_name(file_name, suffix, i + 1))
            else:
                self.__assert_file_exists(directory, self.__get_file_name(file_name, suffix))

    def _assert_output_files_exist(self, file_name: str, suffix: str):
        """
        Asserts that output files, which should be created by a command, exist.
        """
        self._assert_files_exist(self.output_dir, file_name, suffix)

    def _validate_output_files(self):
        """
        May be overridden by subclasses in order to check if certain output files have been created.
        """
        self.__assert_model_files_exist()
        self.__assert_evaluation_files_exist()
        self.__assert_parameter_files_exist()
        self.__assert_prediction_files_exist()
        self.__assert_prediction_characteristic_files_exist()
        self.__assert_data_characteristic_files_exist()
        self.__assert_model_characteristic_files_exist()
        self.__assert_rule_files_exist()

    def run_cmd(self, expected_output_file_name: str = None):
        """
        Runs a command that has been configured via the builder.

        :param expected_output_file_name: The name of the text file that contains the expected output of the command
        """
        for tmp_dir in self.tmp_dirs:
            makedirs(tmp_dir, exist_ok=True)

        out = self.__run_cmd()

        if self.model_dir is not None:
            out = self.__run_cmd()

        if expected_output_file_name is not None:
            stdout = str(out.stdout).splitlines()
            expected_output_dir = self.expected_output_dir

            if OVERWRITE_EXPECTED_OUTPUT_FILES:
                makedirs(expected_output_dir, exist_ok=True)

            expected_output_file = path.join(expected_output_dir, expected_output_file_name + '.txt')

            if OVERWRITE_EXPECTED_OUTPUT_FILES:
                self.__overwrite_output_file(stdout, expected_output_file)
            else:
                self.__assert_output_files_are_equal(stdout, expected_output_file)

        if not OVERWRITE_EXPECTED_OUTPUT_FILES:
            self._validate_output_files()

        self.__remove_tmp_dirs()

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
        self.parameter_dir = parameter_dir

        if parameter_dir is not None:
            self.args.append('--parameter-dir')
            self.args.append(parameter_dir)
        return self

    def no_data_split(self):
        """
        Configures the rule learner to not use separate training and test data.

        :return: The builder itself
        """
        self.num_folds = 0
        self.current_fold = 0
        self.separate_train_test_sets = False
        self.args.append('--data-split')
        self.args.append('none')
        return self

    def cross_validation(self, num_folds: int = 10, current_fold: int = 0):
        """
        Configures the rule learner to use a cross validation.

        :param num_folds:       The total number of folds
        :param current_fold:    The fold to be run or 0, if all folds should be run
        :return:                The builder itself
        """
        self.num_folds = num_folds
        self.current_fold = current_fold
        self.separate_train_test_sets = True
        self.args.append('--data-split')
        self.args.append('cross-validation{num_folds=' + str(num_folds) + ',current_fold=' + str(current_fold) + '}')
        return self

    def sparse_feature_value(self, sparse_feature_value: float = 0.0):
        """
        Configures the value that should be used for sparse elements in the feature matrix.

        :param sparse_feature_value:    The value that should be used for sparse elements in the feature matrix
        :return:                        The builder itself
        """
        self.args.append('--sparse-feature-value')
        self.args.append(str(sparse_feature_value))
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

    def incremental_evaluation(self, incremental_evaluation: bool = True, step_size: int = 50):
        """
        Configures whether the model that is learned by the rule learner should be evaluated repeatedly, using only a
        subset of the rules with increasing size.

        :param incremental_evaluation:  True, if the rule learner should be evaluated incrementally, False otherwise
        :param step_size:               The number of additional rules to be evaluated at each repetition
        :return:                        The builder itself
        """
        self.args.append('--incremental-evaluation')
        value = str(incremental_evaluation).lower()

        if incremental_evaluation:
            value += '{step_size=' + str(step_size) + '}'

        self.args.append(value)
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

    def sparse_output_format(self, sparse: bool = True):
        """
        Configures whether sparse data structures should be used to represent the labels of training examples or not.

        :param sparse:  True, if sparse data structures should be used to represent the labels of training examples,
                        False otherwise
        :return:        The builder itself
        """
        self.args.append('--output-format')
        self.args.append('sparse' if sparse else 'dense')
        return self

    def sparse_prediction_format(self, sparse: bool = True):
        """
        Configures whether sparse data structures should be used to represent predictions or not.

        :param sparse:  True, if sparse data structures should be used to represent predictions, False otherwise
        :return:        The builder itself
        """
        self.args.append('--prediction-format')
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

    def output_sampling(self, output_sampling: str = OUTPUT_SAMPLING_WITHOUT_REPLACEMENT):
        """
        Configures the rule learner to sample from the available outputs.

        :param output_sampling: The name of the sampling method that should be used
        :return:                The builder itself
        """
        self.args.append('--output-sampling')
        self.args.append(output_sampling)
        return self

    def rule_pruning(self, rule_pruning: str = RULE_PRUNING_IREP):
        """
        Configures the rule learner to use a specific method for pruning individual rules.

        :param rule_pruning:    The name of the pruning method that should be used
        :return:                The builder itself
        """
        self.args.append('--rule-pruning')
        self.args.append(rule_pruning)
        return self

    def rule_induction(self, rule_induction=RULE_INDUCTION_TOP_DOWN_GREEDY):
        """
        Configures the rule learner to use a specific algorithm for the induction of individual rules.

        :param rule_induction:  The name of the algorithm that should be used
        :return:                The builder itself
        """
        self.args.append('--rule-induction')
        self.args.append(rule_induction)
        return self

    def sequential_post_optimization(self, sequential_post_optimization: bool = True):
        """
        Configures whether the algorithm should use sequential post-optimization or not.

        :param sequential_post_optimization:    True, if sequential post-optimization should be used, False otherwise
        :return:                                The builder itself
        """
        self.args.append('--sequential-post-optimization')
        self.args.append(str(sequential_post_optimization).lower())
        return self

    def holdout(self, holdout: str = HOLDOUT_RANDOM):
        """
        Configures the algorithm to use a holdout set.

        :param holdout: The name of the sampling method that should be used to create the holdout set
        :return:        The builder itself
        """
        self.args.append('--holdout')
        self.args.append(holdout)
        return self

    def feature_binning(self, feature_binning: str = FEATURE_BINNING_EQUAL_WIDTH):
        """
        Configures the algorithm to use a specific method for feature binning.

        :param feature_binning: The name of the method that should be used for feature binning
        :return:                The builder itself
        """
        self.args.append('--feature-binning')
        self.args.append(feature_binning)
        return self


class IntegrationTests(TestCase, CmdBuilder.AssertionCallback, ABC):
    """
    Defines a series of integration tests for any type of rule learning algorithm.
    """

    def __init__(self,
                 dataset_default: str = DATASET_EMOTIONS,
                 dataset_numerical_sparse: str = DATASET_LANGLOG,
                 dataset_binary: str = DATASET_ENRON,
                 dataset_nominal: str = DATASET_EMOTIONS_NOMINAL,
                 dataset_ordinal: str = DATASET_EMOTIONS_ORDINAL,
                 dataset_single_output: str = DATASET_BREAST_CANCER,
                 dataset_meka: str = DATASET_MEKA,
                 methodName='runTest'):
        """
        :param dataset_default:             The name of the dataset that should be used by default
        :param dataset_numerical_sparse:    The name of a dataset with sparse numerical features
        :param dataset_binary:              The name of a dataset with binary features
        :param dataset_nominal:             The name of a dataset with nominal features
        :param dataset_ordinal:             The name of a dataset with ordinal features
        :param dataset_single_output:       The name of a dataset with a single target variable
        :param dataset_meka:                The name of a dataset in the MEKA format
        """
        super().__init__(methodName)
        self.dataset_default = dataset_default
        self.dataset_numerical_sparse = dataset_numerical_sparse
        self.dataset_binary = dataset_binary
        self.dataset_nominal = dataset_nominal
        self.dataset_ordinal = dataset_ordinal
        self.dataset_single_output = dataset_single_output
        self.dataset_meka = dataset_meka

    def _create_cmd_builder(self, dataset: str = DATASET_EMOTIONS) -> Any:
        """
        Must be implemented by subclasses in order to create an object of type `CmdBuilder` that allows to configure the
        command for running a rule learner.

        :param dataset: The dataset that should be used
        :return:        The object that has been created
        """
        raise NotImplementedError('Method _create_cmd_builder not implemented by test class')

    @classmethod
    def setUpClass(cls):
        if cls is IntegrationTests:
            raise SkipTest(cls.__name__ + ' is an abstract base class')
        if not platform.startswith('linux'):
            raise SkipTest('Integration tests are only supported on Linux')

        super().setUpClass()

    def on_assertion_failure(self, message: str):
        self.fail(message)

    def test_single_output(self):
        """
        Tests the evaluation of the rule learning algorithm when predicting for a single output.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_single_output) \
            .print_evaluation()
        builder.run_cmd('single-output')

    def test_sparse_feature_value(self):
        """
        Tests the training of the rule learning algorithm when using a custom value for the sparse elements in the
        feature matrix.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_numerical_sparse) \
            .sparse_feature_value(1.0)
        builder.run_cmd('sparse-feature-value')

    def test_meka_format(self):
        """
        Tests the evaluation of the rule learning algorithm when using the MEKA data format.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_meka) \
            .print_evaluation(False)
        builder.run_cmd('meka-format')

    def test_evaluation_no_data_split(self):
        """
        Tests the evaluation of the rule learning algorithm when not using a split of the dataset into training and test
        data.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .no_data_split() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        builder.run_cmd('evaluation_no-data-split')

    def test_evaluation_train_test(self):
        """
        Tests the evaluation of the rule learning algorithm when using a predefined split of the dataset into training
        and test data.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        builder.run_cmd('evaluation_train-test')

    def test_evaluation_train_test_predefined(self):
        """
        Tests the evaluation of the rule learning algorithm when using a predefined split of the dataset into training
        and test data, as provided by separate files.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default + '-predefined') \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        builder.run_cmd('evaluation_train-test-predefined')

    def test_evaluation_cross_validation(self):
        """
        Tests the evaluation of the rule learning algorithm when using a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        builder.run_cmd('evaluation_cross-validation')

    def test_evaluation_cross_validation_predefined(self):
        """
        Tests the evaluation of the rule learning algorithm when using predefined splits of the dataset into individual
        cross validation folds, as provided by separate files.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default + '-predefined') \
            .cross_validation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        builder.run_cmd('evaluation_cross-validation-predefined')

    def test_evaluation_single_fold(self):
        """
        Tests the evaluation of the rule learning algorithm when using a single fold of a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation(current_fold=1) \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        builder.run_cmd('evaluation_single-fold')

    def test_evaluation_training_data(self):
        """
        Tests the evaluation of the rule learning algorithm on the training data.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .evaluate_training_data() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        builder.run_cmd('evaluation_training-data')

    def test_evaluation_incremental(self):
        """
        Tests the repeated evaluation of the model that is learned by a rule learning algorithm, using subsets of the
        induced rules with increasing size.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .incremental_evaluation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        builder.run_cmd('evaluation_incremental')

    def test_model_persistence_train_test(self):
        """
        Tests the functionality to store models and load them afterward when using a split of the dataset into training
        and test data.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .set_model_dir()
        builder.run_cmd('model-persistence_train-test')

    def test_model_persistence_cross_validation(self):
        """
        Tests the functionality to store models and load them afterward when using a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation() \
            .set_model_dir()
        builder.run_cmd('model-persistence_cross-validation')

    def test_model_persistence_single_fold(self):
        """
        Tests the functionality to store models and load them afterward when using a single fold of a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation(current_fold=1) \
            .set_model_dir()
        builder.run_cmd('model-persistence_single-fold')

    def test_predictions_train_test(self):
        """
        Tests the functionality to store the predictions of the rule learning algorithm when using a split of the
        dataset into training and test data.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_predictions() \
            .store_predictions()
        builder.run_cmd('predictions_train-test')

    def test_predictions_cross_validation(self):
        """
        Tests the functionality to store the predictions of the rule learning algorithm when using a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_predictions() \
            .store_predictions()
        builder.run_cmd('predictions_cross-validation')

    def test_predictions_single_fold(self):
        """
        Tests the functionality to store the predictions of the rule learning algorithm when using a single fold of a
        cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_predictions() \
            .store_predictions()
        builder.run_cmd('predictions_single-fold')

    def test_predictions_training_data(self):
        """
        Tests the functionality to store the predictions of the rule learning algorithm for the training data.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .evaluate_training_data() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_predictions() \
            .store_predictions()
        builder.run_cmd('predictions_training-data')

    def test_prediction_characteristics_train_test(self):
        """
        Tests the functionality to store the prediction characteristics of the rule learning algorithm when using a
        split of the dataset into training and test data.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_prediction_characteristics() \
            .store_prediction_characteristics()
        builder.run_cmd('prediction-characteristics_train-test')

    def test_prediction_characteristics_cross_validation(self):
        """
        Tests the functionality to store the prediction characteristics of the rule learning algorithm when using a
        cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_prediction_characteristics() \
            .store_prediction_characteristics()
        builder.run_cmd('prediction-characteristics_cross-validation')

    def test_prediction_characteristics_single_fold(self):
        """
        Tests the functionality to store the prediction characteristics of the rule learning algorithm when using a
        single fold of a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_prediction_characteristics() \
            .store_prediction_characteristics()
        builder.run_cmd('prediction-characteristics_single-fold')

    def test_prediction_characteristics_training_data(self):
        """
        Tests the functionality to store the prediction characteristics of the rule learning algorithm for the training
        data.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .evaluate_training_data() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_prediction_characteristics() \
            .store_prediction_characteristics()
        builder.run_cmd('prediction-characteristics_training-data')

    def test_data_characteristics_train_test(self):
        """
        Tests the functionality to store the characteristics of the data used for training by the rule learning
        algorithm when using a split of the dataset into training and test data.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_data_characteristics() \
            .store_data_characteristics()
        builder.run_cmd('data-characteristics_train-test')

    def test_data_characteristics_cross_validation(self):
        """
        Tests the functionality to store the characteristics of the data used for training by the rule learning
        algorithm when using a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_data_characteristics() \
            .store_data_characteristics()
        builder.run_cmd('data-characteristics_cross-validation')

    def test_data_characteristics_single_fold(self):
        """
        Tests the functionality to store the characteristics of the data used for training by the rule learning
        algorithm when using a single fold of a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_data_characteristics() \
            .store_data_characteristics()
        builder.run_cmd('data-characteristics_single-fold')

    def test_model_characteristics_train_test(self):
        """
        Tests the functionality to store the characteristics of models when using a split of the dataset into training
        and test data.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_model_characteristics() \
            .store_model_characteristics()
        builder.run_cmd('model-characteristics_train-test')

    def test_model_characteristics_cross_validation(self):
        """
        Tests the functionality to store the characteristics of models when using a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_model_characteristics() \
            .store_model_characteristics()
        builder.run_cmd('model-characteristics_cross-validation')

    def test_model_characteristics_single_fold(self):
        """
        Tests the functionality to store the characteristics of models when using a single fold of a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_model_characteristics() \
            .store_model_characteristics()
        builder.run_cmd('model-characteristics_single-fold')

    def test_rules_train_test(self):
        """
        Tests the functionality to store textual representations of the rules in a model when using a split of the
        dataset into training and test data.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_rules() \
            .store_rules()
        builder.run_cmd('rules_train-test')

    def test_rules_cross_validation(self):
        """
        Tests the functionality to store textual representations of the rules in a model when using a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_rules() \
            .store_rules()
        builder.run_cmd('rules_cross-validation')

    def test_rules_single_fold(self):
        """
        Tests the functionality to store textual representations of the rules in a model when using a single fold of a
        cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_rules() \
            .store_rules()
        builder.run_cmd('rules_single-fold')

    def test_numeric_features_dense(self):
        """
        Tests the rule learning algorithm on a dataset with numerical features when using a dense feature
        representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_numerical_sparse) \
            .sparse_feature_format(False)
        builder.run_cmd('numeric-features-dense')

    def test_numeric_features_sparse(self):
        """
        Tests the rule learning algorithm on a dataset with numerical features when using a sparse feature
        representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_numerical_sparse) \
            .sparse_feature_format()
        builder.run_cmd('numeric-features-sparse')

    def test_binary_features_dense(self):
        """
        Tests the rule learning algorithm on a dataset with binary features when using a dense feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_binary) \
            .sparse_feature_format(False)
        builder.run_cmd('binary-features-dense')

    def test_binary_features_sparse(self):
        """
        Tests the rule learning algorithm on a dataset with binary features when using a sparse feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_binary) \
            .sparse_feature_format()
        builder.run_cmd('binary-features-sparse')

    def test_nominal_features_dense(self):
        """
        Tests the rule learning algorithm on a dataset with nominal features when using a dense feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_nominal) \
            .sparse_feature_format(False)
        builder.run_cmd('nominal-features-dense')

    def test_nominal_features_sparse(self):
        """
        Tests the rule learning algorithm on a dataset with nominal features when using a sparse feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_nominal) \
            .sparse_feature_format()
        builder.run_cmd('nominal-features-sparse')

    def test_ordinal_features_dense(self):
        """
        Tests the rule learning algorithm on a dataset with ordinal features when using a dense feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_ordinal) \
            .sparse_feature_format(False)
        builder.run_cmd('ordinal-features-dense')

    def test_ordinal_features_sparse(self):
        """
        Tests the rule learning algorithm on a dataset with ordinal features when using a sparse feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_ordinal) \
            .sparse_feature_format()
        builder.run_cmd('ordinal-features-sparse')

    def test_output_format_dense(self):
        """
        Tests the rule learning algorithm when using a dense output representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .sparse_output_format(False)
        builder.run_cmd('output-format-dense')

    def test_output_format_sparse(self):
        """
        Tests the rule learning algorithm when using a sparse output representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .sparse_output_format()
        builder.run_cmd('output-format-sparse')

    def test_prediction_format_dense(self):
        """
        Tests the rule learning algorithm when using a dense representation of predictions.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .sparse_prediction_format(False) \
            .print_predictions()
        builder.run_cmd('prediction-format-dense')

    def test_prediction_format_sparse(self):
        """
        Tests the rule learning algorithm when using a sparse representation of predictions.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .sparse_prediction_format() \
            .print_predictions()
        builder.run_cmd('prediction-format-sparse')

    def test_parameters_train_test(self):
        """
        Tests the functionality to configure the rule learning algorithm according to parameter settings that are loaded
        from input files when using a split of the dataset into training and test data.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_model_characteristics() \
            .print_parameters() \
            .store_parameters() \
            .set_output_dir() \
            .set_parameter_dir()
        builder.run_cmd('parameters_train-test')

    def test_parameters_cross_validation(self):
        """
        Tests the functionality to configure the rule learning algorithm according to parameter settings that are loaded
        from input files when using a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_model_characteristics() \
            .print_parameters() \
            .store_parameters() \
            .set_output_dir() \
            .set_parameter_dir()
        builder.run_cmd('parameters_cross-validation')

    def test_parameters_single_fold(self):
        """
        Tests the functionality to configure the rule learning algorithm according to parameter settings that are loaded
        from input files when using a single fold of a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_model_characteristics() \
            .print_parameters() \
            .store_parameters() \
            .set_output_dir() \
            .set_parameter_dir()
        builder.run_cmd('parameters_single-fold')

    def test_instance_sampling_no(self):
        """
        Tests the rule learning algorithm when not using a method to sample from the available training examples.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .instance_sampling(INSTANCE_SAMPLING_NO)
        builder.run_cmd('instance-sampling-no')

    def test_instance_sampling_with_replacement(self):
        """
        Tests the rule learning algorithm when using a method to sample from the available training examples with
        replacement.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .instance_sampling(INSTANCE_SAMPLING_WITH_REPLACEMENT)
        builder.run_cmd('instance-sampling-with-replacement')

    def test_instance_sampling_without_replacement(self):
        """
        Tests the rule learning algorithm when using a method to sample from the available training examples without
        replacement.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .instance_sampling(INSTANCE_SAMPLING_WITHOUT_REPLACEMENT)
        builder.run_cmd('instance-sampling-without-replacement')

    def test_feature_sampling_no(self):
        """
        Tests the rule learning algorithm when not using a method to sample from the available features.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .feature_sampling(FEATURE_SAMPLING_NO)
        builder.run_cmd('feature-sampling-no')

    def test_feature_sampling_without_replacement(self):
        """
        Tests the rule learning algorithm when using a method to sample from the available features without replacement.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .feature_sampling(FEATURE_SAMPLING_WITHOUT_REPLACEMENT)
        builder.run_cmd('feature-sampling-without-replacement')

    def test_output_sampling_no(self):
        """
        Tests the rule learning algorithm when not using a method to sample from the available outputs.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .output_sampling(OUTPUT_SAMPLING_NO)
        builder.run_cmd('output-sampling-no')

    def test_output_sampling_round_robin(self):
        """
        Tests the rule learning algorithm when using a method that samples single outputs in a round-robin fashion.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .output_sampling(OUTPUT_SAMPLING_ROUND_ROBIN)
        builder.run_cmd('output-sampling-round-robin')

    def test_output_sampling_without_replacement(self):
        """
        Tests the rule learning algorithm when using a method to sample from the available outputs without replacement.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .output_sampling(OUTPUT_SAMPLING_WITHOUT_REPLACEMENT)
        builder.run_cmd('output-sampling-without-replacement')

    def test_pruning_no(self):
        """
        Tests the rule learning algorithm when not using a pruning method.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .rule_pruning(RULE_PRUNING_NO)
        builder.run_cmd('pruning-no')

    def test_pruning_irep(self):
        """
        Tests the rule learning algorithm when using the IREP pruning method.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .instance_sampling() \
            .rule_pruning(RULE_PRUNING_IREP)
        builder.run_cmd('pruning-irep')

    def test_rule_induction_top_down_beam_search(self):
        """
        Tests the rule learning algorithm when using a top-down beam search.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .rule_induction(RULE_INDUCTION_TOP_DOWN_BEAM_SEARCH)
        builder.run_cmd('rule-induction-top-down-beam-search')

    def test_sequential_post_optimization(self):
        """
        Tests the rule learning algorithm when using sequential post-optimization.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .sequential_post_optimization()
        builder.run_cmd('sequential-post-optimization')

    def test_feature_binning_equal_width_binary_features_dense(self):
        """
        Tests the rule learning algorithm's ability to use equal-width feature binning when applied to a dataset with
        binary features using a dense feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_binary) \
            .feature_binning(FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format(False)
        builder.run_cmd('feature-binning-equal-width_binary-features-dense')

    def test_feature_binning_equal_width_binary_features_sparse(self):
        """
        Tests the rule learning algorithm's ability to use equal-width feature binning when applied to a dataset with
        binary features using a sparse feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_binary) \
            .feature_binning(FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format()
        builder.run_cmd('feature-binning-equal-width_binary-features-sparse')

    def test_feature_binning_equal_width_nominal_features_dense(self):
        """
        Tests the rule learning algorithm's ability to use equal-width feature binning when applied to a dataset with
        nominal features using a dense feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_nominal) \
            .feature_binning(FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format(False)
        builder.run_cmd('feature-binning-equal-width_nominal-features-dense')

    def test_feature_binning_equal_width_nominal_features_sparse(self):
        """
        Tests the rule learning algorithm's ability to use equal-width feature binning when applied to a dataset with
        nominal features using a sparse feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_nominal) \
            .feature_binning(FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format()
        builder.run_cmd('feature-binning-equal-width_nominal-features-sparse')

    def test_feature_binning_equal_width_numerical_features_dense(self):
        """
        Tests the rule learning algorithm's ability to use equal-width feature binning when applied to a dataset with
        numerical features using a dense feature representation.
        """
        builder = self._create_cmd_builder() \
            .feature_binning(FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format(False)
        builder.run_cmd('feature-binning-equal-width_numerical-features-dense')

    def test_feature_binning_equal_width_numerical_features_sparse(self):
        """
        Tests the rule learning algorithm's ability to use equal-width feature binning when applied to a dataset with
        numerical features using a sparse feature representation.
        """
        builder = self._create_cmd_builder() \
            .feature_binning(FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format()
        builder.run_cmd('feature-binning-equal-width_numerical-features-sparse')

    def test_feature_binning_equal_frequency_binary_features_dense(self):
        """
        Tests the rule learning algorithm's ability to use equal-frequency feature binning when applied to a dataset
        with binary features using a dense feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_binary) \
            .feature_binning(FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format(False)
        builder.run_cmd('feature-binning-equal-frequency_binary-features-dense')

    def test_feature_binning_equal_frequency_binary_features_sparse(self):
        """
        Tests the rule learning algorithm's ability to use equal-frequency feature binning when applied to a dataset
        with binary features using a sparse feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_binary) \
            .feature_binning(FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format()
        builder.run_cmd('feature-binning-equal-frequency_binary-features-sparse')

    def test_feature_binning_equal_frequency_nominal_features_dense(self):
        """
        Tests the rule learning algorithm's ability to use equal-frequency feature binning when applied to a dataset
        with nominal features using a dense feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_nominal) \
            .feature_binning(FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format(False)
        builder.run_cmd('feature-binning-equal-frequency_nominal-features-dense')

    def test_feature_binning_equal_frequency_nominal_features_sparse(self):
        """
        Tests the rule learning algorithm's ability to use equal-frequency feature binning when applied to a dataset
        with nominal features using a sparse feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_nominal) \
            .feature_binning(FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format()
        builder.run_cmd('feature-binning-equal-frequency_nominal-features-sparse')

    def test_feature_binning_equal_frequency_numerical_features_dense(self):
        """
        Tests the rule learning algorithm's ability to use equal-frequency feature binning when applied to a dataset
        with numerical features using a dense feature representation.
        """
        builder = self._create_cmd_builder() \
            .feature_binning(FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format(False)
        builder.run_cmd('feature-binning-equal-frequency_numerical-features-dense')

    def test_feature_binning_equal_frequency_numerical_features_sparse(self):
        """
        Tests the rule learning algorithm's ability to use equal-frequency feature binning when applied to a dataset
        with numerical features using a sparse feature representation.
        """
        builder = self._create_cmd_builder() \
            .feature_binning(FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format()
        builder.run_cmd('feature-binning-equal-frequency_numerical-features-sparse')
