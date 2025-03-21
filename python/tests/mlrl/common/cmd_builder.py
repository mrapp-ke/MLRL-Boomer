"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
import re
import shutil
import subprocess

from abc import ABC, abstractmethod
from functools import reduce
from os import environ, makedirs, path
from typing import List, Optional

from mlrl.testbed.io import ENCODING_UTF8

ENV_OVERWRITE_OUTPUT_FILES = 'OVERWRITE_OUTPUT_FILES'

DIR_RES = path.join('python', 'tests', 'res')

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
        :param expected_output_dir:     The path to the directory that contains the file with the expected output
        :param model_file_name:         The name of files storing models that have been saved to disk (without suffix)
        :param runnable_module_name:    The fully qualified name of the runnable to be invoked by the 'testbed' program
        :param runnable_class_name:     The class name of the runnable to be invoked by the 'testbed' program
        :param data_dir:                The path to the directory that stores the dataset files
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

        :param directory:   The path to the directory where the file should be located
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

    @staticmethod
    def __should_overwrite_output_files() -> bool:
        value = environ.get(ENV_OVERWRITE_OUTPUT_FILES, 'false').strip().lower()

        if value == 'true':
            return True
        if value == 'false':
            return False
        raise ValueError('Value of environment variable "' + ENV_OVERWRITE_OUTPUT_FILES + '" must be "true" or '
                         + '"false", but is "' + value + '"')

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

        overwrite_output_files = self.__should_overwrite_output_files()

        if expected_output_file_name is not None:
            stdout = str(out.stdout).splitlines()
            expected_output_dir = self.expected_output_dir

            if overwrite_output_files:
                makedirs(expected_output_dir, exist_ok=True)

            expected_output_file = path.join(expected_output_dir, expected_output_file_name + '.txt')

            if overwrite_output_files:
                self.__overwrite_output_file(stdout, expected_output_file)
            else:
                self.__assert_output_files_are_equal(stdout, expected_output_file)

        if not overwrite_output_files:
            self._validate_output_files()

        self.__remove_tmp_dirs()

    def set_output_dir(self, output_dir: Optional[str] = DIR_RESULTS):
        """
        Configures the rule learner to store output files in a given directory.

        :param output_dir:  The path to the directory where output files should be stored
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

        :param model_dir:   The path to the directory where models should be stored
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

        :param parameter_dir:   The path to the directory, where parameter settings are stored
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
