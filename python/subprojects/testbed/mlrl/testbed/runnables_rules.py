"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for programs that can be configured via command line arguments.
"""

from argparse import ArgumentError, ArgumentParser
from typing import Dict, List, Optional, Set

from sklearn.base import ClassifierMixin as SkLearnClassifierMixin, RegressorMixin as SkLearnRegressorMixin

from mlrl.common.config.parameters import Parameter
from mlrl.common.learners import RuleLearner, SparsePolicy

from mlrl.testbed.experiments import Experiment, SkLearnExperiment
from mlrl.testbed.experiments.input.dataset.splitters import DatasetSplitter
from mlrl.testbed.experiments.output.characteristics.model.profile import RuleModelCharacteristicsProfile
from mlrl.testbed.experiments.output.model_text.profile import RuleModelProfile
from mlrl.testbed.experiments.output.probability_calibration import IsotonicJointProbabilityCalibrationModelExtractor, \
    ProbabilityCalibrationModelWriter
from mlrl.testbed.experiments.output.probability_calibration.profile import MarginalProbabilityCalibrationModelProfile
from mlrl.testbed.experiments.output.sinks import CsvFileSink, LogSink
from mlrl.testbed.experiments.output.writer import OutputWriter
from mlrl.testbed.experiments.prediction import IncrementalPredictor
from mlrl.testbed.experiments.prediction_type import PredictionType
from mlrl.testbed.experiments.problem_domain import ClassificationProblem, ProblemDomain, RegressionProblem
from mlrl.testbed.experiments.problem_domain_sklearn import SkLearnProblem
from mlrl.testbed.profiles.profile import Profile
from mlrl.testbed.runnables_sklearn import SkLearnRunnable
from mlrl.testbed.util.format import OPTION_DECIMALS

from mlrl.util.format import format_enum_values
from mlrl.util.options import BooleanOption, parse_param_and_options
from mlrl.util.validation import assert_greater, assert_greater_or_equal


class RuleLearnerRunnable(SkLearnRunnable):
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

    PARAM_PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL = '--print-joint-probability-calibration-model'

    PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {OPTION_DECIMALS},
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_JOINT_PROBABILITY_CALIBRATION_MODEL = '--store-joint-probability-calibration-model'

    STORE_JOINT_PROBABILITY_CALIBRATION_MODEL_VALUES = PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL_VALUES

    PARAM_FEATURE_FORMAT = '--feature-format'

    PARAM_SPARSE_FEATURE_VALUE = '--sparse-feature-value'

    def __init__(self, classifier_type: Optional[type], classifier_config_type: Optional[type],
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
        self.classifier_type = classifier_type
        self.classifier_config_type = classifier_config_type
        self.classifier_parameters = classifier_parameters
        self.regressor_type = regressor_type
        self.regressor_config_type = regressor_config_type
        self.regressor_parameters = regressor_parameters

    def get_profiles(self) -> List[Profile]:
        """
        See :func:`mlrl.testbed.runnables.Runnable.get_profiles`
        """
        return super().get_profiles() + [
            RuleModelProfile(),
            RuleModelCharacteristicsProfile(),
            MarginalProbabilityCalibrationModelProfile()
        ]

    def __create_config_type_and_parameters(self, problem_domain: ProblemDomain):
        if isinstance(problem_domain, ClassificationProblem):
            config_type = self.classifier_config_type
            parameters = self.classifier_parameters
        elif isinstance(problem_domain, RegressionProblem):
            config_type = self.regressor_config_type
            parameters = self.regressor_parameters
        else:
            config_type = None
            parameters = None

        if config_type and parameters:
            return config_type, parameters
        raise RuntimeError('The machine learning algorithm does not support ' + problem_domain.problem_name
                           + ' problems')

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

    def configure_arguments(self, parser: ArgumentParser):
        """
        See :func:`mlrl.testbed.runnables.Runnable.configure_arguments`
        """
        super().configure_arguments(parser)
        parser.add_argument(self.PARAM_INCREMENTAL_EVALUATION,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether models should be evaluated repeatedly, using only a subset of the induced '
                            + 'rules with increasing size, or not. Must be one of ' + format_enum_values(BooleanOption)
                            + '. For additional options refer to the documentation.')
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
        problem_domain = self._create_problem_domain(parser.parse_known_args()[0])
        config_type, parameters = self.__create_config_type_and_parameters(problem_domain)
        self.__configure_argument_parser(parser, config_type, parameters)

    def _create_experiment(self, args, dataset_splitter: DatasetSplitter) -> Experiment:
        kwargs = {RuleLearner.KWARG_SPARSE_FEATURE_VALUE: args.sparse_feature_value}
        problem_domain = self._create_problem_domain(args, fit_kwargs=kwargs, predict_kwargs=kwargs)
        experiment = SkLearnExperiment(problem_domain=problem_domain, dataset_splitter=dataset_splitter)
        experiment.add_post_training_output_writers(*filter(lambda listener: listener is not None, [
            self._create_joint_probability_calibration_model_writer(args),
        ]))
        return experiment

    def create_classifier(self, args) -> Optional[SkLearnClassifierMixin]:
        """
        See :func:`mlrl.testbed.runnables.Runnable.create_classifier`
        """
        classifier_type = self.classifier_type

        if classifier_type:
            kwargs = self.__create_kwargs_from_parameters(self.classifier_parameters, args)
            return classifier_type(**kwargs)
        return None

    def create_regressor(self, args) -> Optional[SkLearnRegressorMixin]:
        """
        See :func:`mlrl.testbed.runnables_sklearn.SkLearnRunnable.create_regressor`
        """
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

        if (not value and args.print_all) or value == BooleanOption.TRUE.value:
            sinks.append(LogSink(options))

        value, options = parse_param_and_options(self.PARAM_STORE_JOINT_PROBABILITY_CALIBRATION_MODEL,
                                                 args.store_joint_probability_calibration_model,
                                                 self.STORE_JOINT_PROBABILITY_CALIBRATION_MODEL_VALUES)

        if ((not value and args.store_all) or value == BooleanOption.TRUE.value) and args.output_dir:
            sinks.append(
                CsvFileSink(directory=args.output_dir, create_directory=args.create_output_dir, options=options))

        if sinks:
            return ProbabilityCalibrationModelWriter(IsotonicJointProbabilityCalibrationModelExtractor(),
                                                     exit_on_error=args.exit_on_error).add_sinks(*sinks)
        return None

    def _create_predictor_factory(self, args, prediction_type: PredictionType) -> SkLearnProblem.PredictorFactory:
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

            def predictor_factory():
                return IncrementalPredictor(prediction_type, min_size=min_size, max_size=max_size, step_size=step_size)

            return predictor_factory

        return super()._create_predictor_factory(args, prediction_type)
