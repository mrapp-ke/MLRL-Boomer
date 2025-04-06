"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for programs that can be configured via command line arguments.
"""
import logging as log
import sys

from abc import ABC, abstractmethod
from argparse import ArgumentError, ArgumentParser
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Set

from sklearn.base import BaseEstimator as SkLearnBaseEstimator, ClassifierMixin as SkLearnClassifierMixin, \
    RegressorMixin as SkLearnRegressorMixin

from mlrl.common.config.options import BooleanOption, parse_param_and_options
from mlrl.common.config.parameters import NONE, Parameter
from mlrl.common.cython.validation import assert_greater, assert_greater_or_equal, assert_less, assert_less_or_equal
from mlrl.common.learners import RuleLearner, SparsePolicy
from mlrl.common.package_info import PythonPackageInfo
from mlrl.common.util.format import format_dict_keys, format_enum_values, format_iterable

from mlrl.testbed.data_splitting import CrossValidationSplitter, DataSet, DataSplitter, NoSplitter, TrainTestSplitter
from mlrl.testbed.experiment import Experiment
from mlrl.testbed.experiments.input.preprocessors import OneHotEncoder, Preprocessor
from mlrl.testbed.experiments.output.characteristics.data import DataCharacteristics, DataCharacteristicsWriter, \
    OutputCharacteristics, PredictionCharacteristicsWriter
from mlrl.testbed.experiments.output.characteristics.model import ModelCharacteristicsWriter, \
    RuleModelCharacteristicsExtractor
from mlrl.testbed.experiments.output.evaluation import ClassificationEvaluationDataExtractor, EvaluationResult, \
    EvaluationWriter, RankingEvaluationDataExtractor, RegressionEvaluationDataExtractor
from mlrl.testbed.experiments.output.label_vectors import LabelVectors, LabelVectorSetExtractor, LabelVectorWriter
from mlrl.testbed.experiments.output.parameters import ParameterWriter
from mlrl.testbed.experiments.output.predictions import PredictionWriter
from mlrl.testbed.experiments.output.sinks import CsvFileSink, LogSink, TextFileSink
from mlrl.testbed.experiments.output.writer import OutputWriter
from mlrl.testbed.experiments.prediction import GlobalPredictor, IncrementalPredictor, Predictor
from mlrl.testbed.experiments.prediction_type import PredictionType
from mlrl.testbed.experiments.problem_type import ProblemType
from mlrl.testbed.models import OPTION_DECIMALS_BODY, OPTION_DECIMALS_HEAD, OPTION_PRINT_BODIES, \
    OPTION_PRINT_FEATURE_NAMES, OPTION_PRINT_HEADS, OPTION_PRINT_NOMINAL_VALUES, OPTION_PRINT_OUTPUT_NAMES, \
    ModelWriter
from mlrl.testbed.package_info import get_package_info as get_testbed_package_info
from mlrl.testbed.parameters import CsvParameterLoader, ParameterLoader
from mlrl.testbed.persistence import ModelLoader, ModelSaver
from mlrl.testbed.probability_calibration import JointProbabilityCalibrationModelWriter, \
    MarginalProbabilityCalibrationModelWriter
from mlrl.testbed.util.format import OPTION_DECIMALS, OPTION_PERCENTAGE, format_table
from mlrl.testbed.util.io import clear_directory

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

            if year:
                result += ' ' + year

            authors = self.authors

            if authors:
                result += ' ' + format_iterable(authors)

            return ('Copyright (c)' if result else '') + result

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
            return 'used by ' + format_iterable(parent_packages) if parent_packages else ''

        def __format_package_info(self) -> str:
            rows = []
            python_packages = self.all_python_packages

            for i, python_package in enumerate(sorted(self.__collect_python_packages(python_packages))):
                rows.append(['' if i > 0 else 'Python packages:', python_package, ''])

            if python_packages:
                rows.append(['', '', ''])

            dependencies = self.__collect_dependencies(python_packages)

            for i, dependency in enumerate(sorted(dependencies.keys())):
                parent_packages = self.__format_parent_packages(dependencies[dependency])
                rows.append(['' if i > 0 else 'Dependencies:', dependency, parent_packages])

            if dependencies:
                rows.append(['', '', ''])

            cpp_libraries = self.__collect_cpp_libraries(python_packages)

            for i, cpp_library in enumerate(sorted(cpp_libraries.keys())):
                parent_packages = self.__format_parent_packages(cpp_libraries[cpp_library])
                rows.append(['' if i > 0 else 'Shared libraries:', cpp_library, parent_packages])

            if cpp_libraries:
                rows.append(['', '', ''])

            build_options = self.__collect_build_options(python_packages)

            for i, build_option in enumerate(sorted(build_options.keys())):
                parent_libraries = self.__format_parent_packages(build_options[build_option])
                rows.append(['' if i > 0 else 'Build options:', build_option, parent_libraries])

            if build_options:
                rows.append(['', '', ''])

            hardware_resources = self.__collect_hardware_resources(python_packages)

            for i, hardware_resource in enumerate(sorted(hardware_resources.keys())):
                for j, info in enumerate(sorted(hardware_resources[hardware_resource])):
                    rows.append(['' if i > 0 else 'Hardware resources:', '' if j > 0 else hardware_resource, info])

            return format_table(rows) if rows else ''

        def __str__(self) -> str:
            result = self.name + ' ' + self.version
            formatted_copyright = self.__format_copyright()

            if formatted_copyright:
                result += '\n\n' + formatted_copyright

            formatted_package_info = self.__format_package_info()

            if formatted_package_info:
                result += '\n\n' + formatted_package_info

            return result

    @staticmethod
    def __get_version(program_info: Optional[ProgramInfo]) -> str:
        """
        May be overridden by subclasses in order to provide information about the program's version.

        :return: A string that provides information about the program's version
        """
        if program_info:
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

        if program_info:
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
            :param output_dir: The path to the output directory from which the files should be deleted
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
            EvaluationResult.OPTION_ENABLE_ALL, EvaluationResult.OPTION_HAMMING_LOSS,
            EvaluationResult.OPTION_HAMMING_ACCURACY, EvaluationResult.OPTION_SUBSET_ZERO_ONE_LOSS,
            EvaluationResult.OPTION_SUBSET_ACCURACY, EvaluationResult.OPTION_MICRO_PRECISION,
            EvaluationResult.OPTION_MICRO_RECALL, EvaluationResult.OPTION_MICRO_F1,
            EvaluationResult.OPTION_MICRO_JACCARD, EvaluationResult.OPTION_MACRO_PRECISION,
            EvaluationResult.OPTION_MACRO_RECALL, EvaluationResult.OPTION_MACRO_F1,
            EvaluationResult.OPTION_MACRO_JACCARD, EvaluationResult.OPTION_EXAMPLE_WISE_PRECISION,
            EvaluationResult.OPTION_EXAMPLE_WISE_RECALL, EvaluationResult.OPTION_EXAMPLE_WISE_F1,
            EvaluationResult.OPTION_EXAMPLE_WISE_JACCARD, EvaluationResult.OPTION_ACCURACY,
            EvaluationResult.OPTION_ZERO_ONE_LOSS, EvaluationResult.OPTION_PRECISION, EvaluationResult.OPTION_RECALL,
            EvaluationResult.OPTION_F1, EvaluationResult.OPTION_JACCARD, EvaluationResult.OPTION_MEAN_ABSOLUTE_ERROR,
            EvaluationResult.OPTION_MEAN_SQUARED_ERROR, EvaluationResult.OPTION_MEDIAN_ABSOLUTE_ERROR,
            EvaluationResult.OPTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR, EvaluationResult.OPTION_RANK_LOSS,
            EvaluationResult.OPTION_COVERAGE_ERROR, EvaluationResult.OPTION_LABEL_RANKING_AVERAGE_PRECISION,
            EvaluationResult.OPTION_DISCOUNTED_CUMULATIVE_GAIN,
            EvaluationResult.OPTION_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN, OPTION_DECIMALS, OPTION_PERCENTAGE
        },
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_EVALUATION = '--store-evaluation'

    STORE_EVALUATION_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {
            EvaluationResult.OPTION_ENABLE_ALL, EvaluationResult.OPTION_HAMMING_LOSS,
            EvaluationResult.OPTION_HAMMING_ACCURACY, EvaluationResult.OPTION_SUBSET_ZERO_ONE_LOSS,
            EvaluationResult.OPTION_SUBSET_ACCURACY, EvaluationResult.OPTION_MICRO_PRECISION,
            EvaluationResult.OPTION_MICRO_RECALL, EvaluationResult.OPTION_MICRO_F1,
            EvaluationResult.OPTION_MICRO_JACCARD, EvaluationResult.OPTION_MACRO_PRECISION,
            EvaluationResult.OPTION_MACRO_RECALL, EvaluationResult.OPTION_MACRO_F1,
            EvaluationResult.OPTION_MACRO_JACCARD, EvaluationResult.OPTION_EXAMPLE_WISE_PRECISION,
            EvaluationResult.OPTION_EXAMPLE_WISE_RECALL, EvaluationResult.OPTION_EXAMPLE_WISE_F1,
            EvaluationResult.OPTION_EXAMPLE_WISE_JACCARD, EvaluationResult.OPTION_ACCURACY,
            EvaluationResult.OPTION_ZERO_ONE_LOSS, EvaluationResult.OPTION_PRECISION, EvaluationResult.OPTION_RECALL,
            EvaluationResult.OPTION_F1, EvaluationResult.OPTION_JACCARD, EvaluationResult.OPTION_MEAN_ABSOLUTE_ERROR,
            EvaluationResult.OPTION_MEAN_SQUARED_ERROR, EvaluationResult.OPTION_MEDIAN_ABSOLUTE_ERROR,
            EvaluationResult.OPTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR, EvaluationResult.OPTION_RANK_LOSS,
            EvaluationResult.OPTION_COVERAGE_ERROR, EvaluationResult.OPTION_LABEL_RANKING_AVERAGE_PRECISION,
            EvaluationResult.OPTION_DISCOUNTED_CUMULATIVE_GAIN,
            EvaluationResult.OPTION_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN, EvaluationResult.OPTION_TRAINING_TIME,
            EvaluationResult.OPTION_PREDICTION_TIME, OPTION_DECIMALS, OPTION_PERCENTAGE
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
            OutputCharacteristics.OPTION_OUTPUTS, OutputCharacteristics.OPTION_OUTPUT_DENSITY,
            OutputCharacteristics.OPTION_OUTPUT_SPARSITY, OutputCharacteristics.OPTION_LABEL_IMBALANCE_RATIO,
            OutputCharacteristics.OPTION_LABEL_CARDINALITY, OutputCharacteristics.OPTION_DISTINCT_LABEL_VECTORS,
            OPTION_DECIMALS, OPTION_PERCENTAGE
        },
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_PREDICTION_CHARACTERISTICS = '--store-prediction-characteristics'

    STORE_PREDICTION_CHARACTERISTICS_VALUES = PRINT_PREDICTION_CHARACTERISTICS_VALUES

    PARAM_PRINT_DATA_CHARACTERISTICS = '--print-data-characteristics'

    PRINT_DATA_CHARACTERISTICS_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {
            DataCharacteristics.OPTION_EXAMPLES, DataCharacteristics.OPTION_FEATURES,
            DataCharacteristics.OPTION_NUMERICAL_FEATURES, DataCharacteristics.OPTION_NOMINAL_FEATURES,
            DataCharacteristics.OPTION_FEATURE_DENSITY, DataCharacteristics.OPTION_FEATURE_SPARSITY,
            OutputCharacteristics.OPTION_OUTPUTS, OutputCharacteristics.OPTION_OUTPUT_DENSITY,
            OutputCharacteristics.OPTION_OUTPUT_SPARSITY, OutputCharacteristics.OPTION_LABEL_IMBALANCE_RATIO,
            OutputCharacteristics.OPTION_LABEL_CARDINALITY, OutputCharacteristics.OPTION_DISTINCT_LABEL_VECTORS,
            OPTION_DECIMALS, OPTION_PERCENTAGE
        },
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_DATA_CHARACTERISTICS = '--store-data-characteristics'

    STORE_DATA_CHARACTERISTICS_VALUES = PRINT_DATA_CHARACTERISTICS_VALUES

    PARAM_PRINT_LABEL_VECTORS = '--print-label-vectors'

    PRINT_LABEL_VECTORS_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {LabelVectors.OPTION_SPARSE},
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

        if base_learner:
            return base_learner
        raise RuntimeError('The machine learning algorithm "' + self.learner_name + '" does not support '
                           + problem_type.value + ' problems')

    def __create_prediction_type(self, args) -> PredictionType:
        return PredictionType.parse(self.PARAM_PREDICTION_TYPE, args.prediction_type)

    @staticmethod
    def __create_preprocessor(args) -> Optional[Preprocessor]:
        if args.one_hot_encoding:
            return OneHotEncoder()
        return None

    def __create_data_splitter(self, args) -> DataSplitter:
        data_set = DataSet(data_dir=args.data_dir,
                           data_set_name=args.dataset,
                           use_one_hot_encoding=args.one_hot_encoding)
        preprocessor = self.__create_preprocessor(args)
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
                                           preprocessor,
                                           num_folds=num_folds,
                                           current_fold=current_fold - 1,
                                           random_state=random_state)
        if value == self.DATA_SPLIT_TRAIN_TEST:
            test_size = options.get_float(self.OPTION_TEST_SIZE, 0.33)
            assert_greater(self.OPTION_TEST_SIZE, test_size, 0)
            assert_less(self.OPTION_TEST_SIZE, test_size, 1)
            random_state = int(args.random_state) if args.random_state else 1
            assert_greater_or_equal(self.PARAM_RANDOM_STATE, random_state, 1)
            return TrainTestSplitter(data_set, preprocessor, test_size=test_size, random_state=random_state)

        return NoSplitter(data_set, preprocessor)

    @staticmethod
    def __create_pre_execution_hook(args, data_splitter: DataSplitter) -> Optional[Experiment.ExecutionHook]:
        output_dir = args.output_dir
        current_fold = data_splitter.current_fold if isinstance(data_splitter, CrossValidationSplitter) else -1
        return LearnerRunnable.ClearOutputDirHook(output_dir) if output_dir and current_fold < 0 else None

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
                            help='The path to the directory where the data set files are located.')
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
        parser.add_argument('--model-load-dir',
                            type=str,
                            help='The path to the directory from which models should be loaded.')
        parser.add_argument('--model-save-dir',
                            type=str,
                            help='The path to the directory to which models should be saved.')
        parser.add_argument('--parameter-load-dir',
                            type=str,
                            help='The path to the directory from which parameter to be used by the algorith should be '
                            + 'loaded.')
        parser.add_argument('--parameter-save-dir',
                            type=str,
                            help='The path to the directory where configuration files, which specify the parameters to '
                            + 'be used by the algorithm, are located.')
        parser.add_argument(self.PARAM_OUTPUT_DIR,
                            type=str,
                            help='The path to the directory where experimental results should be saved.')
        parser.add_argument('--print-parameters',
                            type=BooleanOption.parse,
                            default=False,
                            help='Whether the parameter setting should be printed on the console or not. Must be one '
                            + 'of ' + format_enum_values(BooleanOption) + '.')
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
        data_splitter = self.__create_data_splitter(args)
        pre_execution_hook = self.__create_pre_execution_hook(args, data_splitter)
        pre_training_output_writers = self._create_pre_training_output_writers(args)
        post_training_output_writers = self._create_post_training_output_writers(args)
        prediction_output_writers = self._create_prediction_output_writers(args, problem_type, prediction_type)
        train_predictor = self._create_train_predictor(args, prediction_type) if prediction_output_writers else None
        test_predictor = self._create_test_predictor(args, prediction_type) if prediction_output_writers else None
        parameter_loader = self._create_parameter_loader(args)
        model_loader = self._create_model_loader(args)
        model_saver = self._create_model_saver(args)
        experiment = self._create_experiment(args,
                                             problem_type=problem_type,
                                             base_learner=base_learner,
                                             learner_name=self.learner_name,
                                             data_splitter=data_splitter,
                                             train_predictor=train_predictor,
                                             test_predictor=test_predictor,
                                             pre_training_output_writers=pre_training_output_writers,
                                             post_training_output_writers=post_training_output_writers,
                                             prediction_output_writers=prediction_output_writers,
                                             pre_execution_hook=pre_execution_hook,
                                             parameter_loader=parameter_loader,
                                             model_loader=model_loader,
                                             model_saver=model_saver)
        experiment.run()

    # pylint: disable=unused-argument
    def _create_experiment(self, args, problem_type: ProblemType, base_learner: SkLearnBaseEstimator, learner_name: str,
                           data_splitter: DataSplitter, pre_training_output_writers: List[OutputWriter],
                           post_training_output_writers: List[OutputWriter],
                           prediction_output_writers: List[OutputWriter],
                           pre_execution_hook: Optional[Experiment.ExecutionHook], train_predictor: Optional[Predictor],
                           test_predictor: Optional[Predictor], parameter_loader: Optional[ParameterLoader],
                           model_loader: Optional[ModelLoader], model_saver: Optional[ModelSaver]) -> Experiment:
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
        :param prediction_output_writers:       A list that contains all output writers to be invoked each time
                                                predictions have been obtained from a model
        :param pre_execution_hook:              An operation that should be executed before the experiment
        :param train_predictor:                 The `Predictor` to be used for obtaining predictions for the training
                                                data or None, if no such predictions should be obtained
        :param test_predictor:                  The `Predictor` to be used for obtaining predictions for the test data
                                                or None, if no such predictions should be obtained
        :param parameter_loader:                The `ParameterLoader` that should be used to read the parameter settings
        :param model_loader:                    The `ModelLoader` that should be used for loading models
        :param model_saver:                     The `ModelSaver` that should be used for saving models
        :return:                                The `Experiment` that has been created
        """
        return Experiment(problem_type=problem_type,
                          base_learner=base_learner,
                          learner_name=learner_name,
                          data_splitter=data_splitter,
                          pre_training_output_writers=pre_training_output_writers,
                          post_training_output_writers=post_training_output_writers,
                          prediction_output_writers=prediction_output_writers,
                          pre_execution_hook=pre_execution_hook,
                          train_predictor=train_predictor,
                          test_predictor=test_predictor,
                          parameter_loader=parameter_loader,
                          model_loader=model_loader,
                          model_saver=model_saver)

    def _create_pre_training_output_writers(self, args) -> List[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter`s that should be invoked before training a
        model.

        :param args:    The command line arguments
        :return:        A list that contains the `OutputWriter`s that have been created
        """
        output_writers = []
        output_writer = self._create_data_characteristics_writer(args)

        if output_writer:
            output_writers.append(output_writer)

        output_writer = self._create_parameter_writer(args)

        if output_writer:
            output_writers.append(output_writer)

        return output_writers

    def _create_post_training_output_writers(self, args) -> List[OutputWriter]:
        """
        May be overridden by subclasses in order to create the output writers that should be invoked after training a
        model.

        :param args:    The command line arguments
        :return:        A list that contains the output writers that have been created
        """
        output_writers = []
        output_writer = self._create_label_vector_writer(args)

        if output_writer:
            output_writers.append(output_writer)

        return output_writers

    def _create_prediction_output_writers(self, args, problem_type: ProblemType,
                                          prediction_type: PredictionType) -> List[OutputWriter]:
        """
        May be overridden by subclasses in order to create the output writers that should be invoked each time
        predictions have been obtained from a model.

        :param args:            The command line arguments
        :param problem_type:    The type of the machine learning problem
        :param prediction_type: The type of the predictions that should be obtained
        :return:                A list that contains the output writers that have been created
        """
        output_writers = []
        output_writer = self._create_evaluation_writer(args, problem_type, prediction_type)

        if output_writer:
            output_writers.append(output_writer)

        output_writer = self._create_prediction_writer(args)

        if output_writer:
            output_writers.append(output_writer)

        output_writer = self._create_prediction_characteristics_writer(args)

        if output_writer:
            output_writers.append(output_writer)

        return output_writers

    def _create_model_loader(self, args) -> Optional[ModelLoader]:
        """
        May be overridden by subclasses in order to create the `ModelLoader` that should be used for loading models.

        :param args:    The command line arguments
        :return:        The `ModelLoader` that has been created
        """
        model_load_dir = args.model_load_dir
        return ModelLoader(model_load_dir) if model_load_dir else None

    def _create_model_saver(self, args) -> Optional[ModelSaver]:
        """
        May be overridden by subclasses in order to create the `ModelSaver` that should be used for saving models.

        :param args:    The command line arguments
        :return:        The `ModelSaver` that has been created
        """
        model_save_dir = args.model_save_dir
        return ModelSaver(model_save_dir) if model_save_dir else None

    def _create_train_predictor(self, args, prediction_type: PredictionType) -> Optional[Predictor]:
        """
        May be overridden by subclasses in order to create the `Predictor` that should be used for obtaining predictions
        for the training data from a previously trained model.

        :param args:            The command line arguments
        :param prediction_type: The type of the predictions to be obtained
        :return:                The `Predictor` that has been created or None, if no predictions should be obtained for
                                the training data
        """
        if args.evaluate_training_data:
            return self._create_predictor(args, prediction_type)
        return None

    def _create_test_predictor(self, args, prediction_type: PredictionType) -> Optional[Predictor]:
        """
        May be overridden by subclasses in order to create the `Predictor` that should be used for obtaining predictions
        for the test data from a previously trained model.

        :param args:            The command line arguments
        :param prediction_type: The type of the predictions to be obtained
        :return:                The `Predictor` that has been created or None, if no predictions should be obtained for
                                the test data
        """
        return self._create_predictor(args, prediction_type)

    # pylint: disable=unused-argument
    def _create_predictor(self, args, prediction_type: PredictionType) -> Predictor:
        """
        May be overridden by subclasses in order to create the `Predictor` that should be used for obtaining predictions
        from a previously trained model.

        :param args:            The command line arguments
        :param prediction_type: The type of the predictions to be obtained
        :return:                The `Predictor` that has been created
        """
        return GlobalPredictor(prediction_type)

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
            sinks.append(LogSink(options))

        value, options = parse_param_and_options(self.PARAM_STORE_EVALUATION, args.store_evaluation,
                                                 self.STORE_EVALUATION_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir:
            sinks.append(CsvFileSink(args.output_dir, options=options))

        if sinks:
            if problem_type == ProblemType.REGRESSION:
                extractor = RegressionEvaluationDataExtractor()
            elif prediction_type in {PredictionType.SCORES, PredictionType.PROBABILITIES}:
                extractor = RankingEvaluationDataExtractor()
            else:
                extractor = ClassificationEvaluationDataExtractor()

            return EvaluationWriter(extractor).add_sinks(*sinks)

        return None

    def _create_parameter_loader(self, args) -> Optional[ParameterLoader]:
        """
        May be overridden by subclasses in order to create the `ParameterLoader` that should be used for loading
        parameter settings.

        :param args:    The command line arguments
        :return:        The `ParameterLoader` that has been created
        """
        parameter_load_dir = args.parameter_load_dir
        return CsvParameterLoader(parameter_load_dir) if parameter_load_dir else None

    def _create_parameter_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output parameter
        settings.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []

        if args.print_parameters:
            sinks.append(LogSink())

        if args.parameter_save_dir:
            sinks.append(CsvFileSink(args.parameter_save_dir))

        return ParameterWriter().add_sinks(*sinks) if sinks else None

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
            sinks.append(LogSink(options))

        value, options = parse_param_and_options(self.PARAM_STORE_PREDICTIONS, args.store_predictions,
                                                 self.STORE_PREDICTIONS_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir:
            sinks.append(PredictionWriter.ArffFileSink(args.output_dir, options=options))

        return PredictionWriter().add_sinks(*sinks) if sinks else None

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
            sinks.append(LogSink(options))

        value, options = parse_param_and_options(self.PARAM_STORE_PREDICTION_CHARACTERISTICS,
                                                 args.store_prediction_characteristics,
                                                 self.STORE_PREDICTION_CHARACTERISTICS_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir:
            sinks.append(CsvFileSink(args.output_dir, options=options))

        return PredictionCharacteristicsWriter().add_sinks(*sinks) if sinks else None

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
            sinks.append(LogSink(options))

        value, options = parse_param_and_options(self.PARAM_STORE_DATA_CHARACTERISTICS, args.store_data_characteristics,
                                                 self.STORE_DATA_CHARACTERISTICS_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir:
            sinks.append(CsvFileSink(args.output_dir, options=options))

        return DataCharacteristicsWriter().add_sinks(*sinks) if sinks else None

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
            sinks.append(LogSink(options))

        value, options = parse_param_and_options(self.PARAM_STORE_LABEL_VECTORS, args.store_label_vectors,
                                                 self.STORE_LABEL_VECTORS_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir:
            sinks.append(CsvFileSink(args.output_dir, options=options))

        return LabelVectorWriter(LabelVectorSetExtractor()).add_sinks(*sinks) if sinks else None

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

        if config_type and parameters:
            return config_type, parameters
        raise RuntimeError('The machine learning algorithm "' + self.learner_name + '" does not support '
                           + problem_type.value + ' problems')

    @staticmethod
    def __configure_argument_parser(parser: ArgumentParser, config_type: type, parameters: Set[Parameter]):
        """
        Configure an `ArgumentParser` by taking into account a given set of parameters.

        :param parser:      The `ArgumentParser` to be configured
        :param config_type: The type of the configuration that should support the parameters
        :param parameters:  A set that contains the parameters to be taken into account
        """
        for parameter in parameters:
            try:
                parameter.add_to_argument_parser(parser, config_type)
            except ArgumentError:
                # Argument has already been added, that's okay
                pass

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
        self.__configure_argument_parser(parser, config_type, parameters)

    def _create_experiment(self, args, problem_type: ProblemType, base_learner: SkLearnBaseEstimator, learner_name: str,
                           data_splitter: DataSplitter, pre_training_output_writers: List[OutputWriter],
                           post_training_output_writers: List[OutputWriter],
                           prediction_output_writers: List[OutputWriter],
                           pre_execution_hook: Optional[Experiment.ExecutionHook], train_predictor: Optional[Predictor],
                           test_predictor: Optional[Predictor], parameter_loader: Optional[ParameterLoader],
                           model_loader: Optional[ModelLoader], model_saver: Optional[ModelSaver]) -> Experiment:
        kwargs = {RuleLearner.KWARG_SPARSE_FEATURE_VALUE: args.sparse_feature_value}
        return Experiment(problem_type=problem_type,
                          base_learner=base_learner,
                          learner_name=learner_name,
                          data_splitter=data_splitter,
                          pre_training_output_writers=pre_training_output_writers,
                          post_training_output_writers=post_training_output_writers,
                          prediction_output_writers=prediction_output_writers,
                          pre_execution_hook=pre_execution_hook,
                          train_predictor=train_predictor,
                          test_predictor=test_predictor,
                          parameter_loader=parameter_loader,
                          model_loader=model_loader,
                          model_saver=model_saver,
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
        kwargs = {}
        args_dict = vars(args)

        for parameter in parameters:
            parameter_name = parameter.name

            if parameter_name in args_dict:
                kwargs[parameter_name] = args_dict[parameter_name]

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
            sinks.append(LogSink(options))

        value, options = parse_param_and_options(self.PARAM_STORE_RULES, args.store_rules, self.STORE_RULES_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir:
            sinks.append(TextFileSink(args.output_dir, options=options))

        return ModelWriter().add_sinks(*sinks) if sinks else None

    def _create_model_characteristics_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output the
        characteristics of models.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []

        if args.print_model_characteristics:
            sinks.append(LogSink())

        if args.store_model_characteristics and args.output_dir:
            sinks.append(CsvFileSink(args.output_dir))

        return ModelCharacteristicsWriter(RuleModelCharacteristicsExtractor()).add_sinks(*sinks) if sinks else None

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
            sinks.append(LogSink(options))

        value, options = parse_param_and_options(self.PARAM_STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL,
                                                 args.store_marginal_probability_calibration_model,
                                                 self.STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir:
            sinks.append(CsvFileSink(args.output_dir, options=options))

        return MarginalProbabilityCalibrationModelWriter().add_sinks(*sinks) if sinks else None

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
            sinks.append(LogSink(options))

        value, options = parse_param_and_options(self.PARAM_STORE_JOINT_PROBABILITY_CALIBRATION_MODEL,
                                                 args.store_joint_probability_calibration_model,
                                                 self.STORE_JOINT_PROBABILITY_CALIBRATION_MODEL_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir:
            sinks.append(CsvFileSink(args.output_dir, options=options))

        return JointProbabilityCalibrationModelWriter().add_sinks(*sinks) if sinks else None

    def _create_predictor(self, args, prediction_type: PredictionType) -> Predictor:
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
            return IncrementalPredictor(prediction_type, min_size=min_size, max_size=max_size, step_size=step_size)

        return super()._create_predictor(args, prediction_type)

    def _create_post_training_output_writers(self, args) -> List[OutputWriter]:
        output_writers = super()._create_post_training_output_writers(args)
        output_writer = self._create_model_writer(args)

        if output_writer:
            output_writers.append(output_writer)

        output_writer = self._create_model_characteristics_writer(args)

        if output_writer:
            output_writers.append(output_writer)

        output_writer = self._create_marginal_probability_calibration_model_writer(args)

        if output_writer:
            output_writers.append(output_writer)

        output_writer = self._create_joint_probability_calibration_model_writer(args)

        if output_writer:
            output_writers.append(output_writer)

        return output_writers
