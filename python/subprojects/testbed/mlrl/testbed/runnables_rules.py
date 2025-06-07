"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for programs that can be configured via command line arguments.
"""
from argparse import Namespace
from typing import Optional, Set

from sklearn.base import ClassifierMixin as SkLearnClassifierMixin, RegressorMixin as SkLearnRegressorMixin

from mlrl.common.config.parameters import Parameter
from mlrl.common.learners import RuleLearner, SparsePolicy

from mlrl.testbed.experiments import Experiment, SkLearnExperiment
from mlrl.testbed.experiments.input.dataset.splitters import DatasetSplitter
from mlrl.testbed.experiments.output.characteristics.model.extension import RuleModelCharacteristicsExtension
from mlrl.testbed.experiments.output.model_text.extension import RuleModelAsTextExtension
from mlrl.testbed.experiments.output.probability_calibration.extension import \
    JointProbabilityCalibrationModelExtension, MarginalProbabilityCalibrationModelExtension
from mlrl.testbed.experiments.prediction import IncrementalPredictor
from mlrl.testbed.experiments.prediction.predictor import Predictor
from mlrl.testbed.experiments.prediction_type import PredictionType
from mlrl.testbed.experiments.problem_domain import ClassificationProblem, ProblemDomain, RegressionProblem
from mlrl.testbed.experiments.problem_domain_sklearn import SkLearnProblem
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.runnables_sklearn import SkLearnRunnable

from mlrl.util.cli import Argument, BoolArgument, CommandLineInterface, FloatArgument, SetArgument
from mlrl.util.validation import assert_greater, assert_greater_or_equal

OPTION_MIN_SIZE = 'min_size'

OPTION_MAX_SIZE = 'max_size'

OPTION_STEP_SIZE = 'step_size'


class RuleLearnerRunnable(SkLearnRunnable):
    """
    A base class for all programs that perform an experiment that involves training and evaluation of a rule learner.
    """

    class IncrementalPredictorFactory(SkLearnRunnable.GlobalPredictorFactory):
        """
        Allow to create instances of type `Predictor` that obtain incremental predictions from a model repeatedly.
        """

        def __init__(self, prediction_type: PredictionType, min_size: int, max_size: int, step_size: int):
            """
            :param prediction_type: The type of the predictions to be obtained
            """
            super().__init__(prediction_type)
            self.min_size = min_size
            self.max_size = max_size
            self.step_size = step_size

        def create(self) -> Predictor:
            """
            See :func:`from mlrl.testbed.experiments.problem_domain_sklearn.SkLearnProblem.PredictorFactory.create`
            """
            return IncrementalPredictor(self.prediction_type,
                                        min_size=self.min_size,
                                        max_size=self.max_size,
                                        step_size=self.step_size)

    class IncrementalPredictionExtension(Extension):
        """
        An extension that configures the functionality to obtain incremental predictions.
        """

        INCREMENTAL_EVALUATION = BoolArgument(
            '--incremental-evaluation',
            default=False,
            description='Whether models should be evaluated repeatedly, using only a subset of the induced rules with '
            + 'increasing size, or not.',
            true_options={OPTION_MIN_SIZE, OPTION_MAX_SIZE, OPTION_STEP_SIZE},
        )

        def _get_arguments(self) -> Set[Argument]:
            """
            See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
            """
            return {self.INCREMENTAL_EVALUATION}

        @staticmethod
        def get_predictor_factory(args: Namespace, prediction_type: PredictionType) -> SkLearnProblem.PredictorFactory:
            """
             Returns the `SkLearnProblem.PredictorFactory` that should be used for obtaining predictions of a specific
             type from a previously trained model according to the configuration.

            :param args:            The command line arguments specified by the user
            :param prediction_type: The type of the predictions
            :return:                The `SkLearnProblem.PredictorFactory` that should be used
            """
            value, options = RuleLearnerRunnable.IncrementalPredictionExtension.INCREMENTAL_EVALUATION.get_value(args)

            if value:
                min_size = options.get_int(OPTION_MIN_SIZE, 0)
                assert_greater_or_equal(OPTION_MIN_SIZE, min_size, 0)
                max_size = options.get_int(OPTION_MAX_SIZE, 0)
                if max_size != 0:
                    assert_greater(OPTION_MAX_SIZE, max_size, min_size)
                step_size = options.get_int(OPTION_STEP_SIZE, 1)
                assert_greater_or_equal(OPTION_STEP_SIZE, step_size, 1)
                return RuleLearnerRunnable.IncrementalPredictorFactory(prediction_type,
                                                                       min_size=min_size,
                                                                       max_size=max_size,
                                                                       step_size=step_size)

            return SkLearnRunnable.GlobalPredictorFactory(prediction_type)

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

    def get_extensions(self) -> Set[Extension]:
        """
        See :func:`mlrl.testbed.runnables.Runnable.get_extensions`
        """
        return super().get_extensions() | {
            RuleLearnerRunnable.IncrementalPredictionExtension(),
            RuleModelAsTextExtension(),
            RuleModelCharacteristicsExtension(),
            MarginalProbabilityCalibrationModelExtension(),
            JointProbabilityCalibrationModelExtension()
        }

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

    def configure_arguments(self, cli: CommandLineInterface):
        """
        See :func:`mlrl.testbed.runnables.Runnable.configure_arguments`
        """
        super().configure_arguments(cli)
        cli.add_arguments(
            SetArgument(
                self.PARAM_FEATURE_FORMAT,
                values=SparsePolicy,
                description='The format to be used for the representation of the feature matrix.',
            ),
            FloatArgument(
                self.PARAM_SPARSE_FEATURE_VALUE,
                default=0.0,
                description='The value that should be used for sparse elements in the feature matrix. Does only have '
                + 'an effect if a sparse format is used for the representation of the feature matrix, depending on the '
                + 'argument ' + self.PARAM_FEATURE_FORMAT + '.',
            ),
            SetArgument(
                '--output-format',
                values=SparsePolicy,
                description='The format to be used for the representation of the output matrix.',
            ),
            SetArgument(
                '--prediction-format',
                values=SparsePolicy,
                description='The format to be used for the representation of predictions.',
            ),
        )
        problem_domain = self._create_problem_domain(cli.parse_known_args())
        config_type, parameters = self.__create_config_type_and_parameters(problem_domain)

        for parameter in sorted(parameters, key=lambda param: param.name):
            argument = parameter.as_argument(config_type)

            if argument:
                cli.add_arguments(argument)

    def _create_experiment_builder(self, args: Namespace, dataset_splitter: DatasetSplitter) -> Experiment.Builder:
        kwargs = {RuleLearner.KWARG_SPARSE_FEATURE_VALUE: args.sparse_feature_value}
        problem_domain = self._create_problem_domain(args, fit_kwargs=kwargs, predict_kwargs=kwargs)
        return SkLearnExperiment.Builder(problem_domain=problem_domain, dataset_splitter=dataset_splitter)

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

    def _create_predictor_factory(self, args, prediction_type: PredictionType) -> SkLearnProblem.PredictorFactory:
        return RuleLearnerRunnable.IncrementalPredictionExtension.get_predictor_factory(args, prediction_type)
