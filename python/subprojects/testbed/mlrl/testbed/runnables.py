"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for programs that can be configured via command line arguments.
"""
import logging as log
import sys

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Set

from sklearn.base import BaseEstimator as SkLearnBaseEstimator, ClassifierMixin as SkLearnClassifierMixin, \
    RegressorMixin as SkLearnRegressorMixin

from mlrl.common.config import NONE, Parameter, configure_argument_parser, create_kwargs_from_parameters
from mlrl.common.cython.validation import assert_greater, assert_greater_or_equal, assert_less, assert_less_or_equal
from mlrl.common.format import format_dict_keys, format_enum_values, format_iterable
from mlrl.common.info import PythonPackageInfo
from mlrl.common.options import BooleanOption, parse_param_and_options
from mlrl.common.rule_learners import KWARG_SPARSE_FEATURE_VALUE, SparsePolicy

from mlrl.testbed.characteristics import OPTION_DISTINCT_LABEL_VECTORS, OPTION_LABEL_CARDINALITY, \
    OPTION_LABEL_IMBALANCE_RATIO, OPTION_OUTPUT_DENSITY, OPTION_OUTPUT_SPARSITY, OPTION_OUTPUTS
from mlrl.testbed.data_characteristics import OPTION_EXAMPLES, OPTION_FEATURE_DENSITY, OPTION_FEATURE_SPARSITY, \
    OPTION_FEATURES, OPTION_NOMINAL_FEATURES, OPTION_NUMERICAL_FEATURES, DataCharacteristicsWriter
from mlrl.testbed.data_splitting import CrossValidationSplitter, DataSet, DataSplitter, NoSplitter, TrainTestSplitter
from mlrl.testbed.evaluation import OPTION_ACCURACY, OPTION_COVERAGE_ERROR, OPTION_DISCOUNTED_CUMULATIVE_GAIN, \
    OPTION_ENABLE_ALL, OPTION_EXAMPLE_WISE_F1, OPTION_EXAMPLE_WISE_JACCARD, OPTION_EXAMPLE_WISE_PRECISION, \
    OPTION_EXAMPLE_WISE_RECALL, OPTION_F1, OPTION_HAMMING_ACCURACY, OPTION_HAMMING_LOSS, OPTION_JACCARD, \
    OPTION_LABEL_RANKING_AVERAGE_PRECISION, OPTION_MACRO_F1, OPTION_MACRO_JACCARD, OPTION_MACRO_PRECISION, \
    OPTION_MACRO_RECALL, OPTION_MEAN_ABSOLUTE_ERROR, OPTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR, OPTION_MEAN_SQUARED_ERROR, \
    OPTION_MEDIAN_ABSOLUTE_ERROR, OPTION_MICRO_F1, OPTION_MICRO_JACCARD, OPTION_MICRO_PRECISION, OPTION_MICRO_RECALL, \
    OPTION_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN, OPTION_PRECISION, OPTION_PREDICTION_TIME, OPTION_RANK_LOSS, \
    OPTION_RECALL, OPTION_SUBSET_ACCURACY, OPTION_SUBSET_ZERO_ONE_LOSS, OPTION_TRAINING_TIME, OPTION_ZERO_ONE_LOSS, \
    BinaryEvaluationWriter, EvaluationWriter, RankingEvaluationWriter, RegressionEvaluationWriter
from mlrl.testbed.experiments import Evaluation, Experiment, GlobalEvaluation, IncrementalEvaluation
from mlrl.testbed.format import OPTION_DECIMALS, OPTION_PERCENTAGE, format_table
from mlrl.testbed.info import get_package_info as get_testbed_package_info
from mlrl.testbed.io import clear_directory
from mlrl.testbed.label_vectors import OPTION_SPARSE, LabelVectorSetWriter, LabelVectorWriter
from mlrl.testbed.model_characteristics import ModelCharacteristicsWriter, RuleModelCharacteristicsWriter
from mlrl.testbed.models import OPTION_DECIMALS_BODY, OPTION_DECIMALS_HEAD, OPTION_PRINT_BODIES, \
    OPTION_PRINT_FEATURE_NAMES, OPTION_PRINT_HEADS, OPTION_PRINT_NOMINAL_VALUES, OPTION_PRINT_OUTPUT_NAMES, \
    ModelWriter, RuleModelWriter
from mlrl.testbed.output_writer import OutputWriter
from mlrl.testbed.parameters import ParameterCsvInput, ParameterInput, ParameterWriter
from mlrl.testbed.persistence import ModelPersistence
from mlrl.testbed.prediction_characteristics import PredictionCharacteristicsWriter
from mlrl.testbed.prediction_scope import PredictionType
from mlrl.testbed.predictions import PredictionWriter
from mlrl.testbed.probability_calibration import JointProbabilityCalibrationModelWriter, \
    MarginalProbabilityCalibrationModelWriter
from mlrl.testbed.problem_type import ProblemType

LOG_FORMAT = '%(levelname)s %(message)s'


class LogLevel(Enum):
    """
    Specifies all valid textual representations of log levels.
    """
    DEBUG = 'debug'
    INFO = 'info'
    WARN = 'warn'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'
    FATAL = 'fatal'
    NOTSET = 'notset'

    @staticmethod
    def parse(text: str):
        """
        Parses a given text that represents a log level. If the given text does not represent a valid log level, a
        `ValueError` is raised.

        :param text:    The text to be parsed
        :return:        A log level, depending on the given text
        """
        lower_text = text.lower()
        if lower_text == LogLevel.DEBUG.value:
            return log.DEBUG
        if lower_text == LogLevel.INFO.value:
            return log.INFO
        if lower_text in (LogLevel.WARN.value, LogLevel.WARNING.value):
            return log.WARN
        if lower_text == LogLevel.ERROR.value:
            return log.ERROR
        if lower_text in (LogLevel.CRITICAL.value, LogLevel.FATAL.value):
            return log.CRITICAL
        if lower_text == LogLevel.NOTSET.value:
            return log.NOTSET
        raise ValueError('Invalid log level given. Must be one of ' + format_enum_values(LogLevel) + ', but is "'
                         + str(text) + '".')


class Runnable(ABC):
    """
    A base class for all programs that can be configured via command line arguments.
    """

    @dataclass
    class ProgramInfo:
        """
        Provides information about a program.

        :param name:            A string that specifies the program name
        :param version:         A string that specifies the program version
        :param year:            A string that specifies the year when the program was released
        :param authors:         A set that contains the name of each author of the program
        :param python_packages: A list that contains a `PythonPackageInfo` for each Python package that is used by the
                                program
        """
        name: str
        version: str
        year: Optional[str] = None
        authors: Set[str] = field(default_factory=set)
        python_packages: List[PythonPackageInfo] = field(default_factory=list)

        @property
        def all_python_packages(self) -> List[PythonPackageInfo]:
            """
            A list that contains a `PythonPackageInfo` for each Python package that is used by the program, as well as
            for the testbed package.
            """
            return [get_testbed_package_info()] + self.python_packages

        def __format_copyright(self) -> str:
            result = ''
            year = self.year

            if year is not None:
                result += ' ' + year

            authors = self.authors

            if len(authors) > 0:
                result += ' ' + format_iterable(authors)

            return ('Copyright (c)' if len(result) > 0 else '') + result

        def __collect_python_packages(self, python_packages: Iterable[PythonPackageInfo]) -> Set[str]:
            unique_packages = set()

            for python_package in python_packages:
                unique_packages.add(str(python_package))
                unique_packages.update(self.__collect_python_packages(python_package.python_packages))

            return unique_packages

        def __collect_dependencies(self, python_packages: Iterable[PythonPackageInfo]) -> Dict[str, Set[str]]:
            unique_dependencies = {}

            for python_package in python_packages:
                for dependency in python_package.dependencies:
                    parent_packages = unique_dependencies.setdefault(str(dependency), set())
                    parent_packages.add(python_package.package_name)

                for key, value in self.__collect_dependencies(python_package.python_packages).items():
                    parent_packages = unique_dependencies.setdefault(key, set())
                    parent_packages.update(value)

            return unique_dependencies

        def __collect_cpp_libraries(self, python_packages: Iterable[PythonPackageInfo]) -> Dict[str, Set[str]]:
            unique_libraries = {}

            for python_package in python_packages:
                for cpp_library in python_package.cpp_libraries:
                    parent_packages = unique_libraries.setdefault(str(cpp_library), set())
                    parent_packages.add(python_package.package_name)

                for key, value in self.__collect_cpp_libraries(python_package.python_packages).items():
                    parent_packages = unique_libraries.setdefault(key, set())
                    parent_packages.update(value)

            return unique_libraries

        def __collect_build_options(self, python_packages: Iterable[PythonPackageInfo]) -> Dict[str, Set[str]]:
            unique_build_options = {}

            for python_package in python_packages:
                for cpp_library in python_package.cpp_libraries:
                    for build_option in cpp_library.build_options:
                        parent_libraries = unique_build_options.setdefault(str(build_option), set())
                        parent_libraries.add(cpp_library.library_name)

                for key, value in self.__collect_build_options(python_package.python_packages).items():
                    parent_libraries = unique_build_options.setdefault(key, set())
                    parent_libraries.update(value)

            return unique_build_options

        def __collect_hardware_resources(self, python_packages: Iterable[PythonPackageInfo]) -> Dict[str, Set[str]]:
            unique_hardware_resources = {}

            for python_package in python_packages:
                for cpp_library in python_package.cpp_libraries:
                    for hardware_resource in cpp_library.hardware_resources:
                        info = unique_hardware_resources.setdefault(hardware_resource.resource, set())
                        info.add(hardware_resource.info)

                for key, value in self.__collect_hardware_resources(python_package.python_packages).items():
                    info = unique_hardware_resources.setdefault(key, set())
                    info.update(value)

            return unique_hardware_resources

        @staticmethod
        def __format_parent_packages(parent_packages: Set[str]) -> str:
            return 'used by ' + format_iterable(parent_packages) if len(parent_packages) > 0 else ''

        def __format_package_info(self) -> str:
            rows = []
            python_packages = self.all_python_packages

            for i, python_package in enumerate(sorted(self.__collect_python_packages(python_packages))):
                rows.append(['' if i > 0 else 'Python packages:', python_package, ''])

            if len(python_packages) > 0:
                rows.append(['', '', ''])

            dependencies = self.__collect_dependencies(python_packages)

            for i, dependency in enumerate(sorted(dependencies.keys())):
                parent_packages = self.__format_parent_packages(dependencies[dependency])
                rows.append(['' if i > 0 else 'Dependencies:', dependency, parent_packages])

            if len(dependencies) > 0:
                rows.append(['', '', ''])

            cpp_libraries = self.__collect_cpp_libraries(python_packages)

            for i, cpp_library in enumerate(sorted(cpp_libraries.keys())):
                parent_packages = self.__format_parent_packages(cpp_libraries[cpp_library])
                rows.append(['' if i > 0 else 'Shared libraries:', cpp_library, parent_packages])

            if len(cpp_libraries) > 0:
                rows.append(['', '', ''])

            build_options = self.__collect_build_options(python_packages)

            for i, build_option in enumerate(sorted(build_options.keys())):
                parent_libraries = self.__format_parent_packages(build_options[build_option])
                rows.append(['' if i > 0 else 'Build options:', build_option, parent_libraries])

            if len(build_options) > 0:
                rows.append(['', '', ''])

            hardware_resources = self.__collect_hardware_resources(python_packages)

            for i, hardware_resource in enumerate(sorted(hardware_resources.keys())):
                for j, info in enumerate(sorted(hardware_resources[hardware_resource])):
                    rows.append(['' if i > 0 else 'Hardware resources:', '' if j > 0 else hardware_resource, info])

            return format_table(rows) if len(rows) > 0 else ''

        def __str__(self) -> str:
            result = self.name + ' ' + self.version
            formatted_copyright = self.__format_copyright()

            if len(formatted_copyright) > 0:
                result += '\n\n' + formatted_copyright

            formatted_package_info = self.__format_package_info()

            if len(formatted_package_info) > 0:
                result += '\n\n' + formatted_package_info

            return result

    @staticmethod
    def __get_version(program_info: Optional[ProgramInfo]) -> str:
        """
        May be overridden by subclasses in order to provide information about the program's version.

        :return: A string that provides information about the program's version
        """
        if program_info is not None:
            return str(program_info)

        raise RuntimeError('No information about the program version is available')

    def run(self, args):
        """
        Executes the runnable.

        :param args: The command line arguments
        """
        self.configure_logger(args)
        self._run(args)

    def get_program_info(self) -> Optional[ProgramInfo]:
        """
        May be overridden by subclasses in order to provide information about the program to be printed via the command
        line argument '-v' or '--version'. 

        :return: The `Runnable.ProgramInfo` that has been provided
        """
        return None

    def configure_arguments(self, parser: ArgumentParser):
        """
        May be overridden by subclasses in order to configure the command line arguments of the program.

        :param parser:  An `ArgumentParser` that is used for parsing command line arguments
        """
        # pylint: disable=assignment-from-none
        program_info = self.get_program_info()

        if program_info is not None:
            parser.add_argument('-v',
                                '--version',
                                action='version',
                                version=self.__get_version(program_info),
                                help='Display information about the program\'s version.')

        parser.add_argument('--log-level',
                            type=LogLevel.parse,
                            default=LogLevel.INFO.value,
                            help='The log level to be used. Must be one of ' + format_enum_values(LogLevel) + '.')

    def configure_logger(self, args):
        """
        May be overridden by subclasses in order to configure the logger to be used by the program.

        :param args: The command line arguments
        """
        log_level = args.log_level
        root = log.getLogger()
        root.setLevel(log_level)
        out_handler = log.StreamHandler(sys.stdout)
        out_handler.setLevel(log_level)
        out_handler.setFormatter(log.Formatter(LOG_FORMAT))
        root.addHandler(out_handler)

    @abstractmethod
    def _run(self, args):
        """
        Must be implemented by subclasses in order to run the program.

        :param args: The command line arguments
        """


class LearnerRunnable(Runnable, ABC):
    """
    A base class for all programs that perform an experiment that involves training and evaluation of a machine learning
    algorithm.
    """

    class ClearOutputDirHook(Experiment.ExecutionHook):
        """
        Deletes all files from the output directory before an experiment starts.
        """

        def __init__(self, output_dir: str):
            """
            :param output_dir: The path of the output directory from which the files should be deleted
            """
            self.output_dir = output_dir

        def execute(self):
            """
            See :func:`mlrl.testbed.experiments.Experiment.ExecutionHook.execute`
            """
            clear_directory(self.output_dir)

    PARAM_PROBLEM_TYPE = '--problem-type'

    PARAM_RANDOM_STATE = '--random-state'

    PARAM_DATA_SPLIT = '--data-split'

    DATA_SPLIT_TRAIN_TEST = 'train-test'

    OPTION_TEST_SIZE = 'test_size'

    DATA_SPLIT_CROSS_VALIDATION = 'cross-validation'

    OPTION_NUM_FOLDS = 'num_folds'

    OPTION_CURRENT_FOLD = 'current_fold'

    DATA_SPLIT_VALUES: Dict[str, Set[str]] = {
        NONE: {},
        DATA_SPLIT_TRAIN_TEST: {OPTION_TEST_SIZE},
        DATA_SPLIT_CROSS_VALIDATION: {OPTION_NUM_FOLDS, OPTION_CURRENT_FOLD}
    }

    PARAM_PRINT_EVALUATION = '--print-evaluation'

    PRINT_EVALUATION_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {
            OPTION_ENABLE_ALL, OPTION_HAMMING_LOSS, OPTION_HAMMING_ACCURACY, OPTION_SUBSET_ZERO_ONE_LOSS,
            OPTION_SUBSET_ACCURACY, OPTION_MICRO_PRECISION, OPTION_MICRO_RECALL, OPTION_MICRO_F1, OPTION_MICRO_JACCARD,
            OPTION_MACRO_PRECISION, OPTION_MACRO_RECALL, OPTION_MACRO_F1, OPTION_MACRO_JACCARD,
            OPTION_EXAMPLE_WISE_PRECISION, OPTION_EXAMPLE_WISE_RECALL, OPTION_EXAMPLE_WISE_F1,
            OPTION_EXAMPLE_WISE_JACCARD, OPTION_ACCURACY, OPTION_ZERO_ONE_LOSS, OPTION_PRECISION, OPTION_RECALL,
            OPTION_F1, OPTION_JACCARD, OPTION_MEAN_ABSOLUTE_ERROR, OPTION_MEAN_SQUARED_ERROR,
            OPTION_MEDIAN_ABSOLUTE_ERROR, OPTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR, OPTION_RANK_LOSS,
            OPTION_COVERAGE_ERROR, OPTION_LABEL_RANKING_AVERAGE_PRECISION, OPTION_DISCOUNTED_CUMULATIVE_GAIN,
            OPTION_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN, OPTION_DECIMALS, OPTION_PERCENTAGE
        },
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_EVALUATION = '--store-evaluation'

    STORE_EVALUATION_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {
            OPTION_ENABLE_ALL, OPTION_HAMMING_LOSS, OPTION_HAMMING_ACCURACY, OPTION_SUBSET_ZERO_ONE_LOSS,
            OPTION_SUBSET_ACCURACY, OPTION_MICRO_PRECISION, OPTION_MICRO_RECALL, OPTION_MICRO_F1, OPTION_MICRO_JACCARD,
            OPTION_MACRO_PRECISION, OPTION_MACRO_RECALL, OPTION_MACRO_F1, OPTION_MACRO_JACCARD,
            OPTION_EXAMPLE_WISE_PRECISION, OPTION_EXAMPLE_WISE_RECALL, OPTION_EXAMPLE_WISE_F1,
            OPTION_EXAMPLE_WISE_JACCARD, OPTION_ACCURACY, OPTION_ZERO_ONE_LOSS, OPTION_PRECISION, OPTION_RECALL,
            OPTION_F1, OPTION_JACCARD, OPTION_MEAN_ABSOLUTE_ERROR, OPTION_MEAN_SQUARED_ERROR,
            OPTION_MEDIAN_ABSOLUTE_ERROR, OPTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR, OPTION_RANK_LOSS,
            OPTION_COVERAGE_ERROR, OPTION_LABEL_RANKING_AVERAGE_PRECISION, OPTION_DISCOUNTED_CUMULATIVE_GAIN,
            OPTION_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN, OPTION_TRAINING_TIME, OPTION_PREDICTION_TIME, OPTION_DECIMALS,
            OPTION_PERCENTAGE
        },
        BooleanOption.FALSE.value: {}
    }

    PARAM_PRINT_PREDICTIONS = '--print-predictions'

    PRINT_PREDICTIONS_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {OPTION_DECIMALS},
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_PREDICTIONS = '--store-predictions'

    STORE_PREDICTIONS_VALUES = PRINT_PREDICTIONS_VALUES

    PARAM_PRINT_PREDICTION_CHARACTERISTICS = '--print-prediction-characteristics'

    PRINT_PREDICTION_CHARACTERISTICS_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {
            OPTION_OUTPUTS, OPTION_OUTPUT_DENSITY, OPTION_OUTPUT_SPARSITY, OPTION_LABEL_IMBALANCE_RATIO,
            OPTION_LABEL_CARDINALITY, OPTION_DISTINCT_LABEL_VECTORS, OPTION_DECIMALS, OPTION_PERCENTAGE
        },
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_PREDICTION_CHARACTERISTICS = '--store-prediction-characteristics'

    STORE_PREDICTION_CHARACTERISTICS_VALUES = PRINT_PREDICTION_CHARACTERISTICS_VALUES

    PARAM_PRINT_DATA_CHARACTERISTICS = '--print-data-characteristics'

    PRINT_DATA_CHARACTERISTICS_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {
            OPTION_EXAMPLES, OPTION_FEATURES, OPTION_NUMERICAL_FEATURES, OPTION_NOMINAL_FEATURES,
            OPTION_FEATURE_DENSITY, OPTION_FEATURE_SPARSITY, OPTION_OUTPUTS, OPTION_OUTPUT_DENSITY,
            OPTION_OUTPUT_SPARSITY, OPTION_LABEL_IMBALANCE_RATIO, OPTION_LABEL_CARDINALITY,
            OPTION_DISTINCT_LABEL_VECTORS, OPTION_DECIMALS, OPTION_PERCENTAGE
        },
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_DATA_CHARACTERISTICS = '--store-data-characteristics'

    STORE_DATA_CHARACTERISTICS_VALUES = PRINT_DATA_CHARACTERISTICS_VALUES

    PARAM_PRINT_LABEL_VECTORS = '--print-label-vectors'

    PRINT_LABEL_VECTORS_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {OPTION_SPARSE},
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_LABEL_VECTORS = '--store-label-vectors'

    STORE_LABEL_VECTORS_VALUES = PRINT_LABEL_VECTORS_VALUES

    PARAM_OUTPUT_DIR = '--output-dir'

    PARAM_PREDICTION_TYPE = '--prediction-type'

    def __init__(self, learner_name: str):
        """
        :param learner_name: The name of the learner
        """
        super().__init__()
        self.learner_name = learner_name

    def __create_problem_type(self, args) -> ProblemType:
        return ProblemType.parse(self.PARAM_PROBLEM_TYPE, args.problem_type)

    def __create_base_learner(self, problem_type: ProblemType, args) -> SkLearnBaseEstimator:
        if problem_type == ProblemType.CLASSIFICATION:
            base_learner = self.create_classifier(args)
        elif problem_type == ProblemType.REGRESSION:
            base_learner = self.create_regressor(args)
        else:
            base_learner = None

        if base_learner is not None:
            return base_learner
        raise RuntimeError('The machine learning algorithm "' + self.learner_name + '" does not support '
                           + problem_type.value + ' problems')

    def __create_prediction_type(self, args) -> PredictionType:
        return PredictionType.parse(self.PARAM_PREDICTION_TYPE, args.prediction_type)

    def __create_data_splitter(self, args) -> DataSplitter:
        data_set = DataSet(data_dir=args.data_dir,
                           data_set_name=args.dataset,
                           use_one_hot_encoding=args.one_hot_encoding)
        value, options = parse_param_and_options(self.PARAM_DATA_SPLIT, args.data_split, self.DATA_SPLIT_VALUES)

        if value == self.DATA_SPLIT_CROSS_VALIDATION:
            num_folds = options.get_int(self.OPTION_NUM_FOLDS, 10)
            assert_greater_or_equal(self.OPTION_NUM_FOLDS, num_folds, 2)
            current_fold = options.get_int(self.OPTION_CURRENT_FOLD, 0)
            if current_fold != 0:
                assert_greater_or_equal(self.OPTION_CURRENT_FOLD, current_fold, 1)
                assert_less_or_equal(self.OPTION_CURRENT_FOLD, current_fold, num_folds)
            random_state = int(args.random_state) if args.random_state else 1
            assert_greater_or_equal(self.PARAM_RANDOM_STATE, random_state, 1)
            return CrossValidationSplitter(data_set,
                                           num_folds=num_folds,
                                           current_fold=current_fold - 1,
                                           random_state=random_state)
        if value == self.DATA_SPLIT_TRAIN_TEST:
            test_size = options.get_float(self.OPTION_TEST_SIZE, 0.33)
            assert_greater(self.OPTION_TEST_SIZE, test_size, 0)
            assert_less(self.OPTION_TEST_SIZE, test_size, 1)
            random_state = int(args.random_state) if args.random_state else 1
            assert_greater_or_equal(self.PARAM_RANDOM_STATE, random_state, 1)
            return TrainTestSplitter(data_set, test_size=test_size, random_state=random_state)

        return NoSplitter(data_set)

    @staticmethod
    def __create_pre_execution_hook(args, data_splitter: DataSplitter) -> Optional[Experiment.ExecutionHook]:
        current_fold = data_splitter.current_fold if isinstance(data_splitter, CrossValidationSplitter) else -1
        return None if args.output_dir is None or current_fold >= 0 else LearnerRunnable.ClearOutputDirHook(
            output_dir=args.output_dir)

    def configure_arguments(self, parser: ArgumentParser):
        super().configure_arguments(parser)
        parser.add_argument(self.PARAM_PROBLEM_TYPE,
                            type=str,
                            default=ProblemType.CLASSIFICATION.value,
                            help='The type of the machine learning problem to be solved. Must be one of '
                            + format_enum_values(ProblemType) + '.')
        problem_type = self.__create_problem_type(parser.parse_known_args()[0])
        self.configure_problem_specific_arguments(parser, problem_type)

    # pylint: disable=unused-argument
    def configure_problem_specific_arguments(self, parser: ArgumentParser, problem_type: ProblemType):
        """
        May be overridden by subclasses in order to configure the command line arguments of the program, depending on
        the type of machine learning problem to be solved.

        :param parser:          An `ArgumentParser` that is used for parsing command line arguments
        :param problem_type:    The type of the machine learning problem to be solved
        """
        parser.add_argument(self.PARAM_RANDOM_STATE,
                            type=int,
                            default=None,
                            help='The seed to be used by random number generators. Must be at least 1.')
        parser.add_argument('--data-dir',
                            type=str,
                            required=True,
                            help='The path of the directory where the data set files are located.')
        parser.add_argument('--dataset', type=str, required=True, help='The name of the data set files without suffix.')
        parser.add_argument(self.PARAM_DATA_SPLIT,
                            type=str,
                            default=self.DATA_SPLIT_TRAIN_TEST,
                            help='The strategy to be used for splitting the available data into training and test '
                            + 'sets. Must be one of ' + format_dict_keys(self.DATA_SPLIT_VALUES) + '. For additional '
                            + 'options refer to the documentation.')
        parser.add_argument(self.PARAM_PRINT_EVALUATION,
                            type=str,
                            default=BooleanOption.TRUE.value,
                            help='Whether the evaluation results should be printed on the console or not. Must be one '
                            + 'of ' + format_dict_keys(self.PRINT_EVALUATION_VALUES) + '. For additional options refer '
                            + 'to the documentation.')
        parser.add_argument(self.PARAM_STORE_EVALUATION,
                            type=str,
                            default=BooleanOption.TRUE.value,
                            help='Whether the evaluation results should be written into output files or not. Must be '
                            + 'one of ' + format_dict_keys(self.STORE_EVALUATION_VALUES) + '. Does only have an effect '
                            + 'if the parameter ' + self.PARAM_OUTPUT_DIR + ' is specified. For additional options '
                            + 'refer to the documentation.')
        parser.add_argument('--evaluate-training-data',
                            type=BooleanOption.parse,
                            default=False,
                            help='Whether the models should not only be evaluated on the test data, but also on the '
                            + 'training data. Must be one of ' + format_enum_values(BooleanOption) + '.')
        parser.add_argument(self.PARAM_PRINT_PREDICTION_CHARACTERISTICS,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the characteristics of binary predictions should be printed on the console '
                            + 'or not. Must be one of ' + format_dict_keys(self.PRINT_PREDICTION_CHARACTERISTICS_VALUES)
                            + '. Does only have an effect if the parameter ' + self.PARAM_PREDICTION_TYPE + ' is set '
                            + 'to ' + PredictionType.BINARY.value + '. For additional options refer to the '
                            + 'documentation.')
        parser.add_argument(self.PARAM_STORE_PREDICTION_CHARACTERISTICS,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the characteristics of binary predictions should be written into output '
                            + 'files or not. Must be one of '
                            + format_dict_keys(self.STORE_PREDICTION_CHARACTERISTICS_VALUES) + '. Does only have an '
                            + 'effect if the parameter ' + self.PARAM_PREDICTION_TYPE + ' is set to '
                            + PredictionType.BINARY.value + '. For additional options refer to the documentation.')
        parser.add_argument(self.PARAM_PRINT_DATA_CHARACTERISTICS,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the characteristics of the training data should be printed on the console or '
                            + 'not. Must be one of ' + format_dict_keys(self.PRINT_DATA_CHARACTERISTICS_VALUES) + '. '
                            + 'For additional options refer to the documentation.')
        parser.add_argument(self.PARAM_STORE_DATA_CHARACTERISTICS,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the characteristics of the training data should be written into output files '
                            + 'or not. Must be one of ' + format_dict_keys(self.STORE_DATA_CHARACTERISTICS_VALUES)
                            + '. Does only have an effect if the parameter ' + self.PARAM_OUTPUT_DIR + ' is specified. '
                            + 'For additional options refer to the documentation.')
        parser.add_argument(self.PARAM_PRINT_LABEL_VECTORS,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the unique label vectors contained in the training data should be printed on '
                            + 'the console or not. Must be one of ' + format_dict_keys(self.PRINT_LABEL_VECTORS_VALUES)
                            + '. For additional options refer to the documentation.')
        parser.add_argument(self.PARAM_STORE_LABEL_VECTORS,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the unique label vectors contained in the training data should be written '
                            + 'into output files or not. Must be one of '
                            + format_dict_keys(self.STORE_LABEL_VECTORS_VALUES) + '. Does only have an effect if the '
                            + 'parameter ' + self.PARAM_OUTPUT_DIR + ' is specified. For additional options refer to '
                            + 'the documentation.')
        parser.add_argument('--one-hot-encoding',
                            type=BooleanOption.parse,
                            default=False,
                            help='Whether one-hot-encoding should be used to encode nominal features or not. Must be '
                            + 'one of ' + format_enum_values(BooleanOption) + '.')
        parser.add_argument('--model-dir', type=str, help='The path of the directory where models should be stored.')
        parser.add_argument('--parameter-dir',
                            type=str,
                            help='The path of the directory where configuration files, which specify the parameters to '
                            + 'be used by the algorithm, are located.')
        parser.add_argument(self.PARAM_OUTPUT_DIR,
                            type=str,
                            help='The path of the directory where experimental results should be saved.')
        parser.add_argument('--print-parameters',
                            type=BooleanOption.parse,
                            default=False,
                            help='Whether the parameter setting should be printed on the console or not. Must be one '
                            + 'of ' + format_enum_values(BooleanOption) + '.')
        parser.add_argument('--store-parameters',
                            type=BooleanOption.parse,
                            default=False,
                            help='Whether the parameter setting should be written into output files or not. Must be '
                            + 'one of ' + format_enum_values(BooleanOption) + '. Does only have an effect, if the '
                            + 'parameter ' + self.PARAM_OUTPUT_DIR + ' is specified.')
        parser.add_argument(self.PARAM_PRINT_PREDICTIONS,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether predictions should be printed on the console or not. Must be one of '
                            + format_dict_keys(self.PRINT_PREDICTIONS_VALUES) + '. For additional options refer to the '
                            + 'documentation.')
        parser.add_argument(self.PARAM_STORE_PREDICTIONS,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether predictions should be written into output files or not. Must be one of '
                            + format_dict_keys(self.STORE_PREDICTIONS_VALUES) + '. Does only have an effect, if the '
                            + 'parameter ' + self.PARAM_OUTPUT_DIR + ' is specified. For additional options refer to '
                            + 'the documentation.')
        parser.add_argument(self.PARAM_PREDICTION_TYPE,
                            type=str,
                            default=PredictionType.BINARY.value,
                            help='The type of predictions that should be obtained from the learner. Must be one of '
                            + format_enum_values(PredictionType) + '.')

    def _run(self, args):
        problem_type = self.__create_problem_type(args)
        base_learner = self.__create_base_learner(problem_type, args)
        prediction_type = self.__create_prediction_type(args)
        train_evaluation = self._create_train_evaluation(args, problem_type, prediction_type)
        test_evaluation = self._create_test_evaluation(args, problem_type, prediction_type)
        data_splitter = self.__create_data_splitter(args)
        pre_execution_hook = self.__create_pre_execution_hook(args, data_splitter)
        pre_training_output_writers = self._create_pre_training_output_writers(args)
        post_training_output_writers = self._create_post_training_output_writers(args)
        parameter_input = self._create_parameter_input(args)
        persistence = self._create_persistence(args)
        experiment = self._create_experiment(args,
                                             problem_type=problem_type,
                                             base_learner=base_learner,
                                             learner_name=self.learner_name,
                                             data_splitter=data_splitter,
                                             train_evaluation=train_evaluation,
                                             test_evaluation=test_evaluation,
                                             pre_training_output_writers=pre_training_output_writers,
                                             post_training_output_writers=post_training_output_writers,
                                             pre_execution_hook=pre_execution_hook,
                                             parameter_input=parameter_input,
                                             persistence=persistence)
        experiment.run()

    # pylint: disable=unused-argument
    def _create_experiment(self, args, problem_type: ProblemType, base_learner: SkLearnBaseEstimator, learner_name: str,
                           data_splitter: DataSplitter, pre_training_output_writers: List[OutputWriter],
                           post_training_output_writers: List[OutputWriter],
                           pre_execution_hook: Optional[Experiment.ExecutionHook],
                           train_evaluation: Optional[Evaluation], test_evaluation: Optional[Evaluation],
                           parameter_input: Optional[ParameterInput],
                           persistence: Optional[ModelPersistence]) -> Experiment:
        """
        May be overridden by subclasses in order to create the `Experiment` that should be run.

        :param args:                            The command line arguments
        :param problem_type:                    The type of the machine learning problem
        :param base_learner:                    The machine learning algorithm to be used
        :param learner_name:                    The name of machine learning algorithm
        :param data_splitter:                   The method to be used for splitting the available data into training and
                                                test sets
        :param pre_training_output_writers:     A list that contains all output writers to be invoked before training
        :param post_training_output_writers:    A list that contains all output writers to be invoked after training
        :param pre_execution_hook:              An operation that should be executed before the experiment
        :param train_evaluation:                The method to be used for evaluating the predictions for the training
                                                data or None, if the predictions should not be evaluated
        :param test_evaluation:                 The method to be used for evaluating the predictions for the test data
                                                or None, if the predictions should not be evaluated
        :param parameter_input:                 The input that should be used to read the parameter settings
        :param persistence:                     The `ModelPersistence` that should be used for loading and saving models
        :return:                                The `Experiment` that has been created
        """
        return Experiment(problem_type=problem_type,
                          base_learner=base_learner,
                          learner_name=learner_name,
                          data_splitter=data_splitter,
                          pre_training_output_writers=pre_training_output_writers,
                          post_training_output_writers=post_training_output_writers,
                          pre_execution_hook=pre_execution_hook,
                          train_evaluation=train_evaluation,
                          test_evaluation=test_evaluation,
                          parameter_input=parameter_input,
                          persistence=persistence)

    def _create_pre_training_output_writers(self, args) -> List[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter`s that should be invoked before training a
        model.

        :param args:    The command line arguments
        :return:        A list that contains the `OutputWriter`s that have been created
        """
        output_writers = []
        output_writer = self._create_data_characteristics_writer(args)

        if output_writer is not None:
            output_writers.append(output_writer)

        output_writer = self._create_parameter_writer(args)

        if output_writer is not None:
            output_writers.append(output_writer)

        return output_writers

    def _create_post_training_output_writers(self, args) -> List[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter`s that should be invoked after training a
        model.

        :param args:    The command line arguments
        :return:        A list that contains the `OutputWriters`s that have been created
        """
        output_writers = []
        output_writer = self._create_label_vector_writer(args)

        if output_writer is not None:
            output_writers.append(output_writer)

        return output_writers

    def _create_evaluation_output_writers(self, args, problem_type: ProblemType,
                                          prediction_type: PredictionType) -> List[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter`s that should be invoked after evaluating a
        model.

        :param args:            The command line arguments
        :param problem_type:    The type of the machine learning problem
        :param prediction_type: The type of the predictions
        :return:                A list that contains the `OutputWriter`s that have been created
        """
        output_writers = []
        output_writer = self._create_evaluation_writer(args, problem_type, prediction_type)

        if output_writer is not None:
            output_writers.append(output_writer)

        output_writer = self._create_prediction_writer(args)

        if output_writer is not None:
            output_writers.append(output_writer)

        output_writer = self._create_prediction_characteristics_writer(args)

        if output_writer is not None:
            output_writers.append(output_writer)

        return output_writers

    def _create_persistence(self, args) -> Optional[ModelPersistence]:
        """
        May be overridden by subclasses in order to create the `ModelPersistence` that should be used to save and load
        models.

        :param args:    The command line arguments
        :return:        The `ModelPersistence` that has been created
        """
        return None if args.model_dir is None else ModelPersistence(model_dir=args.model_dir)

    def _create_train_evaluation(self, args, problem_type: ProblemType,
                                 prediction_type: PredictionType) -> Optional[Evaluation]:
        """
        May be overridden by subclasses in order to create the `Evaluation` that should be used for evaluating
        predictions obtained from a previously trained model for the training data.

        :param args:            The command line arguments
        :param problem_type:    The type of the machine learning problem
        :param prediction_type: The type of the predictions to be obtained
        :return:                The `Evaluation` that has been created
        """
        if args.evaluate_training_data:
            output_writers = self._create_evaluation_output_writers(args, problem_type, prediction_type)
        else:
            output_writers = []

        return self._create_evaluation(args, prediction_type, output_writers)

    def _create_test_evaluation(self, args, problem_type: ProblemType,
                                prediction_type: PredictionType) -> Optional[Evaluation]:
        """
        May be overridden by subclasses in order to create the `Evaluation` that should be used for evaluating
        predictions obtained from a previously trained model for the test data.

        :param args:            The command line arguments
        :param problem_type:    The type of the machine learning problem
        :param prediction_type: The type of the predictions to be obtained
        :return:                The `Evaluation` that has been created
        """
        output_writers = self._create_evaluation_output_writers(args, problem_type, prediction_type)
        return self._create_evaluation(args, prediction_type, output_writers)

    # pylint: disable=unused-argument
    def _create_evaluation(self, args, prediction_type: PredictionType,
                           output_writers: List[OutputWriter]) -> Optional[Evaluation]:
        """
        May be overridden by subclasses in order to create the `Evaluation` that should be used for evaluating
        predictions obtained from a previously trained model.

        :param args:            The command line arguments
        :param prediction_type: The type of the predictions to be obtained
        :param output_writers:  A list that contains all output writers to be invoked after predictions have been
                                obtained
        :return:                The `Evaluation` that has been created
        """
        return GlobalEvaluation(prediction_type, output_writers) if len(output_writers) > 0 else None

    def _create_evaluation_writer(self, args, problem_type: ProblemType,
                                  prediction_type: PredictionType) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output evaluation
        results.

        :param args:            The command line arguments
        :param problem_type:    The type of the machine learning problem
        :param prediction_type: The type of the predictions
        :return:                The `OutputWriter` that has been created
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_EVALUATION, args.print_evaluation,
                                                 self.PRINT_EVALUATION_VALUES)

        if value == BooleanOption.TRUE.value:
            sinks.append(EvaluationWriter.LogSink(options=options))

        value, options = parse_param_and_options(self.PARAM_STORE_EVALUATION, args.store_evaluation,
                                                 self.STORE_EVALUATION_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir is not None:
            sinks.append(EvaluationWriter.CsvSink(output_dir=args.output_dir, options=options))

        if len(sinks) == 0:
            return None
        if problem_type == ProblemType.REGRESSION:
            return RegressionEvaluationWriter(sinks)
        if prediction_type in {PredictionType.SCORES, PredictionType.PROBABILITIES}:
            return RankingEvaluationWriter(sinks)
        return BinaryEvaluationWriter(sinks)

    def _create_parameter_input(self, args) -> Optional[ParameterInput]:
        """
        May be overridden by subclasses in order to create the `ParameterInput` that should be used to load parameter
        settings.

        :param args:    The command line arguments
        :return:        The `ParameterInput` that has been created
        """
        return None if args.parameter_dir is None else ParameterCsvInput(input_dir=args.parameter_dir)

    def _create_parameter_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output parameter
        settings.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []

        if args.print_parameters:
            sinks.append(ParameterWriter.LogSink())

        if args.store_parameters and args.parameter_dir is not None:
            sinks.append(ParameterWriter.CsvSink(output_dir=args.parameter_dir))

        return ParameterWriter(sinks) if len(sinks) > 0 else None

    def _create_prediction_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output predictions.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_PREDICTIONS, args.print_predictions,
                                                 self.PRINT_PREDICTIONS_VALUES)

        if value == BooleanOption.TRUE.value:
            sinks.append(PredictionWriter.LogSink(options=options))

        value, options = parse_param_and_options(self.PARAM_STORE_PREDICTIONS, args.store_predictions,
                                                 self.STORE_PREDICTIONS_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir is not None:
            sinks.append(PredictionWriter.ArffSink(output_dir=args.output_dir, options=options))

        return PredictionWriter(sinks) if len(sinks) > 0 else None

    def _create_prediction_characteristics_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output prediction
        characteristics.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_PREDICTION_CHARACTERISTICS,
                                                 args.print_prediction_characteristics,
                                                 self.PRINT_PREDICTION_CHARACTERISTICS_VALUES)

        if value == BooleanOption.TRUE.value:
            sinks.append(PredictionCharacteristicsWriter.LogSink(options=options))

        value, options = parse_param_and_options(self.PARAM_STORE_PREDICTION_CHARACTERISTICS,
                                                 args.store_prediction_characteristics,
                                                 self.STORE_PREDICTION_CHARACTERISTICS_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir is not None:
            sinks.append(PredictionCharacteristicsWriter.CsvSink(output_dir=args.output_dir, options=options))

        return PredictionCharacteristicsWriter(sinks) if len(sinks) > 0 else None

    def _create_data_characteristics_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output data
        characteristics.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_DATA_CHARACTERISTICS, args.print_data_characteristics,
                                                 self.PRINT_DATA_CHARACTERISTICS_VALUES)

        if value == BooleanOption.TRUE.value:
            sinks.append(DataCharacteristicsWriter.LogSink(options=options))

        value, options = parse_param_and_options(self.PARAM_STORE_DATA_CHARACTERISTICS, args.store_data_characteristics,
                                                 self.STORE_DATA_CHARACTERISTICS_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir is not None:
            sinks.append(DataCharacteristicsWriter.CsvSink(output_dir=args.output_dir, options=options))

        return DataCharacteristicsWriter(sinks) if len(sinks) > 0 else None

    def _create_label_vector_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output unique label
        vectors contained in the training data.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_LABEL_VECTORS, args.print_label_vectors,
                                                 self.PRINT_LABEL_VECTORS_VALUES)

        if value == BooleanOption.TRUE.value:
            sinks.append(LabelVectorWriter.LogSink(options=options))

        value, options = parse_param_and_options(self.PARAM_STORE_LABEL_VECTORS, args.store_label_vectors,
                                                 self.STORE_LABEL_VECTORS_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir is not None:
            sinks.append(LabelVectorWriter.CsvSink(output_dir=args.output_dir, options=options))

        return LabelVectorWriter(sinks) if len(sinks) > 0 else None

    @abstractmethod
    def create_classifier(self, args) -> Optional[SkLearnClassifierMixin]:
        """
        Must be implemented by subclasses in order to create a machine learning algorithm that can be applied to
        classification problems.

        :param args:    The command line arguments
        :return:        The learner that has been created or None, if regression problems are not supported
        """

    @abstractmethod
    def create_regressor(self, args) -> Optional[SkLearnRegressorMixin]:
        """
        Must be implemented by subclasses in order to create a machine learning algorithm that can be applied to
        regression problems.

        :param args:    The command line arguments
        :return:        The learner that has been created or None, if regression problems are not supported
        """


class RuleLearnerRunnable(LearnerRunnable):
    """
    A base class for all programs that perform an experiment that involves training and evaluation of a rule learner.
    """

    PARAM_INCREMENTAL_EVALUATION = '--incremental-evaluation'

    OPTION_MIN_SIZE = 'min_size'

    OPTION_MAX_SIZE = 'max_size'

    OPTION_STEP_SIZE = 'step_size'

    INCREMENTAL_EVALUATION_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {OPTION_MIN_SIZE, OPTION_MAX_SIZE, OPTION_STEP_SIZE},
        BooleanOption.FALSE.value: {}
    }

    PARAM_PRINT_RULES = '--print-rules'

    PRINT_RULES_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {
            OPTION_PRINT_FEATURE_NAMES, OPTION_PRINT_OUTPUT_NAMES, OPTION_PRINT_NOMINAL_VALUES, OPTION_PRINT_BODIES,
            OPTION_PRINT_HEADS, OPTION_DECIMALS_BODY, OPTION_DECIMALS_HEAD
        },
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_RULES = '--store-rules'

    STORE_RULES_VALUES = PRINT_RULES_VALUES

    PARAM_PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL = '--print-marginal-probability-calibration-model'

    PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {OPTION_DECIMALS},
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL = '--store-marginal-probability-calibration-model'

    STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES = PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES

    PARAM_PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL = '--print-joint-probability-calibration-model'

    PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {OPTION_DECIMALS},
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_JOINT_PROBABILITY_CALIBRATION_MODEL = '--store-joint-probability-calibration-model'

    STORE_JOINT_PROBABILITY_CALIBRATION_MODEL_VALUES = PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL_VALUES

    PARAM_FEATURE_FORMAT = '--feature-format'

    PARAM_SPARSE_FEATURE_VALUE = '--sparse-feature-value'

    def __init__(self, learner_name: str, classifier_type: Optional[type], classifier_config_type: Optional[type],
                 classifier_parameters: Optional[Set[Parameter]], regressor_type: Optional[type],
                 regressor_config_type: Optional[type], regressor_parameters: Optional[Set[Parameter]]):
        """
        :param classifier_type:         The type of the rule learner to be used in classification problems or None, if
                                        classification problems are not supported
        :param classifier_config_type:  The type of the configuration to be used in classification problems or None, if
                                        classification problems are not supported
        :param classifier_parameters:   A set that contains the parameters that may be supported by the rule learner to
                                        be used in regression problems or None, if regression problems are not supported
        :param regressor_type:          The type of the rule learner to be used in regression problems or None, if
                                        regression problems are not supported
        :param regressor_config_type:   The type of the configuration to be used in regression problems or None, if
                                        regression problems are not supported
        :param regressor_parameters:    A set that contains the parameters that may be supported by the rule learner to
                                        be used in regression problems or None, if regression problems are not supported
        """
        super().__init__(learner_name=learner_name)
        self.classifier_type = classifier_type
        self.classifier_config_type = classifier_config_type
        self.classifier_parameters = classifier_parameters
        self.regressor_type = regressor_type
        self.regressor_config_type = regressor_config_type
        self.regressor_parameters = regressor_parameters

    def __create_config_type_and_parameters(self, problem_type: ProblemType):
        if problem_type == ProblemType.CLASSIFICATION:
            config_type = self.classifier_config_type
            parameters = self.classifier_parameters
        elif problem_type == ProblemType.REGRESSION:
            config_type = self.regressor_config_type
            parameters = self.regressor_parameters
        else:
            config_type = None
            parameters = None

        if config_type is not None and parameters is not None:
            return config_type, parameters
        raise RuntimeError('The machine learning algorithm "' + self.learner_name + '" does not support '
                           + problem_type.value + ' problems')

    def configure_problem_specific_arguments(self, parser: ArgumentParser, problem_type: ProblemType):
        super().configure_problem_specific_arguments(parser, problem_type)
        parser.add_argument(self.PARAM_INCREMENTAL_EVALUATION,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether models should be evaluated repeatedly, using only a subset of the induced '
                            + 'rules with increasing size, or not. Must be one of ' + format_enum_values(BooleanOption)
                            + '. For additional options refer to the documentation.')
        parser.add_argument('--print-model-characteristics',
                            type=BooleanOption.parse,
                            default=BooleanOption.FALSE.value,
                            help='Whether the characteristics of models should be printed on the console or not. Must '
                            + 'be one of ' + format_enum_values(BooleanOption) + '.')
        parser.add_argument('--store-model-characteristics',
                            type=BooleanOption.parse,
                            default=BooleanOption.FALSE.value,
                            help='Whether the characteristics of models should be written into output files or not. '
                            + 'Must be one of ' + format_enum_values(BooleanOption) + '. Does only have an effect if '
                            + 'the parameter ' + self.PARAM_OUTPUT_DIR + ' is specified.')
        parser.add_argument(self.PARAM_PRINT_RULES,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the induced rules should be printed on the console or not. Must be one of '
                            + format_dict_keys(self.PRINT_RULES_VALUES) + '. For additional options refer to the '
                            + 'documentation.')
        parser.add_argument(self.PARAM_STORE_RULES,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the induced rules should be written into a text file or not. Must be one of '
                            + format_dict_keys(self.STORE_RULES_VALUES) + '. Does only have an effect if the parameter '
                            + self.PARAM_OUTPUT_DIR + ' is specified. For additional options refer to the '
                            + 'documentation.')
        parser.add_argument(self.PARAM_PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the model for the calibration of marginal probabilities should be printed on '
                            + 'the console or not. Must be one of ' + format_enum_values(BooleanOption) + '. For '
                            + 'additional options refer to the documentation.')
        parser.add_argument(self.PARAM_STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the model for the calibration of marginal probabilities should be written '
                            + 'into an output file or not. Must be one of ' + format_enum_values(BooleanOption) + '. '
                            + 'Does only have an effect if the parameter ' + self.PARAM_OUTPUT_DIR + ' is specified. '
                            + 'For additional options refer to the documentation.')
        parser.add_argument(self.PARAM_PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the model for the calibration of joint probabilities should be printed on '
                            + 'the console or not. Must be one of ' + format_enum_values(BooleanOption) + '. For '
                            + 'additional options refer to the documentation.')
        parser.add_argument(self.PARAM_STORE_JOINT_PROBABILITY_CALIBRATION_MODEL,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the model for the calibration of joint probabilities should be written into '
                            + 'an output file or not. Must be one of ' + format_enum_values(BooleanOption) + '. Does '
                            + 'only have an effect if the parameter ' + self.PARAM_OUTPUT_DIR + ' is specified. For '
                            + 'additional options refer to the documentation.')
        parser.add_argument(self.PARAM_FEATURE_FORMAT,
                            type=str,
                            default=None,
                            help='The format to be used for the representation of the feature matrix. Must be one of '
                            + format_enum_values(SparsePolicy) + '.')
        parser.add_argument(self.PARAM_SPARSE_FEATURE_VALUE,
                            type=float,
                            default=0.0,
                            help='The value that should be used for sparse elements in the feature matrix. Does only '
                            + 'have an effect if a sparse format is used for the representation of the feature matrix, '
                            + 'depending on the parameter ' + self.PARAM_FEATURE_FORMAT + '.')
        parser.add_argument('--output-format',
                            type=str,
                            default=None,
                            help='The format to be used for the representation of the output matrix. Must be one of '
                            + format_enum_values(SparsePolicy) + '.')
        parser.add_argument('--prediction-format',
                            type=str,
                            default=None,
                            help='The format to be used for the representation of predictions. Must be one of '
                            + format_enum_values(SparsePolicy) + '.')
        config_type, parameters = self.__create_config_type_and_parameters(problem_type)
        configure_argument_parser(parser, config_type, parameters)

    def _create_experiment(self, args, problem_type: ProblemType, base_learner: SkLearnBaseEstimator, learner_name: str,
                           data_splitter: DataSplitter, pre_training_output_writers: List[OutputWriter],
                           post_training_output_writers: List[OutputWriter],
                           pre_execution_hook: Optional[Experiment.ExecutionHook],
                           train_evaluation: Optional[Evaluation], test_evaluation: Optional[Evaluation],
                           parameter_input: Optional[ParameterInput],
                           persistence: Optional[ModelPersistence]) -> Experiment:
        kwargs = {KWARG_SPARSE_FEATURE_VALUE: args.sparse_feature_value}
        return Experiment(problem_type=problem_type,
                          base_learner=base_learner,
                          learner_name=learner_name,
                          data_splitter=data_splitter,
                          pre_training_output_writers=pre_training_output_writers,
                          post_training_output_writers=post_training_output_writers,
                          pre_execution_hook=pre_execution_hook,
                          train_evaluation=train_evaluation,
                          test_evaluation=test_evaluation,
                          parameter_input=parameter_input,
                          persistence=persistence,
                          fit_kwargs=kwargs,
                          predict_kwargs=kwargs)

    def create_classifier(self, args) -> Optional[SkLearnClassifierMixin]:
        classifier_type = self.classifier_type

        if classifier_type:
            kwargs = self.__create_kwargs_from_parameters(self.classifier_parameters, args)
            return classifier_type(**kwargs)
        return None

    def create_regressor(self, args) -> Optional[SkLearnRegressorMixin]:
        regressor_type = self.regressor_type

        if regressor_type:
            kwargs = self.__create_kwargs_from_parameters(self.regressor_parameters, args)
            return regressor_type(**kwargs)
        return None

    @staticmethod
    def __create_kwargs_from_parameters(parameters: Set[Parameter], args):
        kwargs = create_kwargs_from_parameters(args, parameters)
        kwargs['random_state'] = args.random_state
        kwargs['feature_format'] = args.feature_format
        kwargs['output_format'] = args.output_format
        kwargs['prediction_format'] = args.prediction_format
        return kwargs

    def _create_model_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output textual
        representations of models.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_RULES, args.print_rules, self.PRINT_RULES_VALUES)

        if value == BooleanOption.TRUE.value:
            sinks.append(ModelWriter.LogSink(options=options))

        value, options = parse_param_and_options(self.PARAM_STORE_RULES, args.store_rules, self.STORE_RULES_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir is not None:
            sinks.append(ModelWriter.TxtSink(output_dir=args.output_dir, options=options))

        return RuleModelWriter(sinks) if len(sinks) > 0 else None

    def _create_model_characteristics_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output the
        characteristics of models.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []

        if args.print_model_characteristics:
            sinks.append(ModelCharacteristicsWriter.LogSink())

        if args.store_model_characteristics and args.output_dir is not None:
            sinks.append(ModelCharacteristicsWriter.CsvSink(output_dir=args.output_dir))

        return RuleModelCharacteristicsWriter(sinks) if len(sinks) > 0 else None

    def _create_marginal_probability_calibration_model_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output textual
        representations of models for the calibration of marginal probabilities.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL,
                                                 args.print_marginal_probability_calibration_model,
                                                 self.PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES)

        if value == BooleanOption.TRUE.value:
            sinks.append(MarginalProbabilityCalibrationModelWriter.LogSink(options=options))

        value, options = parse_param_and_options(self.PARAM_STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL,
                                                 args.store_marginal_probability_calibration_model,
                                                 self.STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir is not None:
            sinks.append(MarginalProbabilityCalibrationModelWriter.CsvSink(output_dir=args.output_dir, options=options))

        return MarginalProbabilityCalibrationModelWriter(sinks) if len(sinks) > 0 else None

    def _create_joint_probability_calibration_model_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output textual
        representations of models for the calibration of joint probabilities.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL,
                                                 args.print_joint_probability_calibration_model,
                                                 self.PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL_VALUES)

        if value == BooleanOption.TRUE.value:
            sinks.append(JointProbabilityCalibrationModelWriter.LogSink(options=options))

        value, options = parse_param_and_options(self.PARAM_STORE_JOINT_PROBABILITY_CALIBRATION_MODEL,
                                                 args.store_joint_probability_calibration_model,
                                                 self.STORE_JOINT_PROBABILITY_CALIBRATION_MODEL_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir is not None:
            sinks.append(JointProbabilityCalibrationModelWriter.CsvSink(output_dir=args.output_dir, options=options))

        return JointProbabilityCalibrationModelWriter(sinks) if len(sinks) > 0 else None

    def _create_evaluation(self, args, prediction_type: PredictionType,
                           output_writers: List[OutputWriter]) -> Optional[Evaluation]:
        value, options = parse_param_and_options(self.PARAM_INCREMENTAL_EVALUATION, args.incremental_evaluation,
                                                 self.INCREMENTAL_EVALUATION_VALUES)

        if value == BooleanOption.TRUE.value:
            min_size = options.get_int(self.OPTION_MIN_SIZE, 0)
            assert_greater_or_equal(self.OPTION_MIN_SIZE, min_size, 0)
            max_size = options.get_int(self.OPTION_MAX_SIZE, 0)
            if max_size != 0:
                assert_greater(self.OPTION_MAX_SIZE, max_size, min_size)
            step_size = options.get_int(self.OPTION_STEP_SIZE, 1)
            assert_greater_or_equal(self.OPTION_STEP_SIZE, step_size, 1)
            return IncrementalEvaluation(
                prediction_type, output_writers, min_size=min_size, max_size=max_size,
                step_size=step_size) if len(output_writers) > 0 else None

        return super()._create_evaluation(args, prediction_type, output_writers)

    def _create_label_vector_writer(self, args) -> Optional[OutputWriter]:
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_LABEL_VECTORS, args.print_label_vectors,
                                                 self.PRINT_LABEL_VECTORS_VALUES)

        if value == BooleanOption.TRUE.value:
            sinks.append(LabelVectorSetWriter.LogSink(options=options))

        value, options = parse_param_and_options(self.PARAM_STORE_LABEL_VECTORS, args.store_label_vectors,
                                                 self.STORE_LABEL_VECTORS_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir is not None:
            sinks.append(LabelVectorSetWriter.CsvSink(output_dir=args.output_dir, options=options))

        return LabelVectorSetWriter(sinks) if len(sinks) > 0 else None

    def _create_post_training_output_writers(self, args) -> List[OutputWriter]:
        output_writers = super()._create_post_training_output_writers(args)
        output_writer = self._create_model_writer(args)

        if output_writer is not None:
            output_writers.append(output_writer)

        output_writer = self._create_model_characteristics_writer(args)

        if output_writer is not None:
            output_writers.append(output_writer)

        output_writer = self._create_marginal_probability_calibration_model_writer(args)

        if output_writer is not None:
            output_writers.append(output_writer)

        output_writer = self._create_joint_probability_calibration_model_writer(args)

        if output_writer is not None:
            output_writers.append(output_writer)

        return output_writers
