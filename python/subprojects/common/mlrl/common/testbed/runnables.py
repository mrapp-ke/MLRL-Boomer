"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for programs that can be configured via command line arguments.
"""
from argparse import Namespace
from typing import Any, Dict, Optional, Set, Type, override

from sklearn.base import BaseEstimator, ClassifierMixin as SkLearnClassifierMixin, \
    RegressorMixin as SkLearnRegressorMixin

from mlrl.common.config.parameters import Parameter
from mlrl.common.cython.learner import RuleLearnerConfig
from mlrl.common.learners import RuleLearner, SparsePolicy
from mlrl.common.testbed.experiments.experiment import RuleLearnerExperiment
from mlrl.common.testbed.experiments.output.characteristics.model import RuleModelCharacteristicsExtension
from mlrl.common.testbed.experiments.output.label_vectors.extension import LabelVectorSetExtension
from mlrl.common.testbed.experiments.output.model_text import RuleModelAsTextExtension
from mlrl.common.testbed.experiments.prediction.predictor_incremental import IncrementalPredictor

from mlrl.testbed_sklearn.experiments import SkLearnProblem
from mlrl.testbed_sklearn.experiments.prediction.predictor import Predictor
from mlrl.testbed_sklearn.runnables import SkLearnRunnable

from mlrl.testbed.command import Command
from mlrl.testbed.experiments import Experiment
from mlrl.testbed.experiments.meta_data import MetaData
from mlrl.testbed.experiments.prediction_type import PredictionType
from mlrl.testbed.experiments.problem_domain import ClassificationProblem, RegressionProblem
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.cli import Argument, BoolArgument, EnumArgument, FloatArgument
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

        @override
        def create(self) -> Predictor:
            """
            See :func:`from mlrl.testbed_sklearn.experiments.problem_domain.SkLearnProblem.PredictorFactory.create`
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

        @override
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

    class RuleLearnerExtension(Extension):
        """
        An extension that configures the algorithmic parameters of a rule learner.
        """

        FEATURE_FORMAT = EnumArgument(
            '--feature-format',
            enum=SparsePolicy,
            description='The format to be used for the representation of the feature matrix.',
        )

        OUTPUT_FORMAT = EnumArgument(
            '--output-format',
            enum=SparsePolicy,
            description='The format to be used for the representation of the output matrix.',
        )

        PREDICTION_FORMAT = EnumArgument(
            '--prediction-format',
            enum=SparsePolicy,
            description='The format to be used for the representation of predictions.',
        )

        SPARSE_FEATURE_VALUE = FloatArgument(
            '--sparse-feature-value',
            default=0.0,
            description='The value that should be used for sparse elements in the feature matrix. Does only have an '
            + 'effect if a sparse format is used for the representation of the feature matrix, depending '
            + 'on the argument ' + FEATURE_FORMAT.name + '.',
        )

        @override
        def _get_arguments(self) -> Set[Argument]:
            """
            See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
            """
            return {self.FEATURE_FORMAT, self.OUTPUT_FORMAT, self.PREDICTION_FORMAT, self.SPARSE_FEATURE_VALUE}

        @staticmethod
        def get_estimator(args: Namespace, estimator_type: Type[BaseEstimator],
                          parameters: Optional[Set[Parameter]]) -> Any:
            """
            Returns the scikit-learn estimator to be used in an experiment.

            :param args:            The command line arguments specified by the user
            :param estimator_type:  The type of the estimator
            :param parameters:      The algorithmic parameters of the estimator
            """
            feature_format = RuleLearnerRunnable.RuleLearnerExtension.FEATURE_FORMAT.get_value(args)
            output_format = RuleLearnerRunnable.RuleLearnerExtension.OUTPUT_FORMAT.get_value(args)
            prediction_format = RuleLearnerRunnable.RuleLearnerExtension.PREDICTION_FORMAT.get_value(args)
            args_dict = vars(args)
            kwargs = {}

            if parameters:
                for parameter in parameters:
                    parameter_name = parameter.name

                    if parameter_name in args_dict:
                        kwargs[parameter_name] = args_dict[parameter_name]

            kwargs['feature_format'] = feature_format.value if feature_format else None
            kwargs['output_format'] = output_format.value if output_format else None
            kwargs['prediction_format'] = prediction_format.value if prediction_format else None
            return estimator_type(**kwargs)

        @staticmethod
        def get_fit_kwargs(args: Namespace) -> Dict[str, Any]:
            """
            Returns the keyword arguments that should be passed to the estimators `fit` function.

            :param args:    The command line arguments specified by the user
            :return:        A dictionary that stores the keyword arguments
            """
            return {
                RuleLearner.KWARG_SPARSE_FEATURE_VALUE:
                    RuleLearnerRunnable.RuleLearnerExtension.SPARSE_FEATURE_VALUE.get_value(args)
            }

        @staticmethod
        def get_predict_kwargs(args: Namespace) -> Dict[str, Any]:
            """
            Returns the keyword arguments that should be passed to the estimators `predict` function.

            :param args:    The command line arguments specified by the user
            :return:        A dictionary that stores the keyword arguments
            """
            return RuleLearnerRunnable.RuleLearnerExtension.get_fit_kwargs(args)

    def __init__(self, classifier_type: Optional[Type[SkLearnClassifierMixin]],
                 classifier_config_type: Optional[Type[RuleLearnerConfig]],
                 classifier_parameters: Optional[Set[Parameter]], regressor_type: Optional[Type[SkLearnRegressorMixin]],
                 regressor_config_type: Optional[Type[RuleLearnerConfig]],
                 regressor_parameters: Optional[Set[Parameter]]):
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

    @override
    def get_extensions(self) -> Set[Extension]:
        """
        See :func:`mlrl.testbed.runnables.Runnable.get_extensions`
        """
        return super().get_extensions() | {
            RuleLearnerRunnable.IncrementalPredictionExtension(),
            RuleLearnerRunnable.RuleLearnerExtension(),
            RuleModelAsTextExtension(),
            RuleModelCharacteristicsExtension(),
            LabelVectorSetExtension(),
        }

    @override
    def get_algorithmic_arguments(self, known_args: Namespace) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.runnables.Runnable.get_algorithmic_arguments`
        """
        problem_domain = SkLearnRunnable.ProblemDomainExtension.get_problem_domain(known_args, runnable=self)
        config_type = None
        parameters = None

        if isinstance(problem_domain, ClassificationProblem):
            config_type = self.classifier_config_type
            parameters = self.classifier_parameters
        elif isinstance(problem_domain, RegressionProblem):
            config_type = self.regressor_config_type
            parameters = self.regressor_parameters

        if not config_type or not parameters:
            raise RuntimeError('The machine learning algorithm does not support ' + problem_domain.problem_name
                               + ' problems')

        return set(filter(None, map(lambda param: param.as_argument(config_type), parameters)))

    @override
    def create_problem_domain(self, args: Namespace):
        """
        See :func:`mlrl.testbed.experiments.recipe.Recipe.create_problem_domain`
        """
        fit_kwargs = RuleLearnerRunnable.RuleLearnerExtension.get_fit_kwargs(args)
        predict_kwargs = RuleLearnerRunnable.RuleLearnerExtension.get_predict_kwargs(args)
        return SkLearnRunnable.ProblemDomainExtension.get_problem_domain(args,
                                                                         runnable=self,
                                                                         fit_kwargs=fit_kwargs,
                                                                         predict_kwargs=predict_kwargs)

    @override
    def create_experiment_builder(self, args: Namespace, command: Command) -> Experiment.Builder:
        """
        See :func:`mlrl.testbed.experiments.recipe.Recipe.create_experiment_builder`
        """
        meta_data = MetaData(command=command)
        initial_state = ExperimentState(meta_data=meta_data, problem_domain=self.create_problem_domain(args))
        return RuleLearnerExperiment.Builder(initial_state=initial_state,
                                             dataset_splitter=self.create_dataset_splitter(args))

    @override
    def create_classifier(self, args: Namespace) -> Optional[SkLearnClassifierMixin]:
        """
        See :func:`mlrl.testbed.runnables.Runnable.create_classifier`
        """
        classifier_type = self.classifier_type

        if classifier_type:
            return RuleLearnerRunnable.RuleLearnerExtension.get_estimator(args,
                                                                          estimator_type=classifier_type,
                                                                          parameters=self.classifier_parameters)
        return None

    @override
    def create_regressor(self, args: Namespace) -> Optional[SkLearnRegressorMixin]:
        """
        See :func:`mlrl.testbed_sklearn.runnables.SkLearnRunnable.create_regressor`
        """
        regressor_type = self.regressor_type

        if regressor_type:
            return RuleLearnerRunnable.RuleLearnerExtension.get_estimator(args,
                                                                          estimator_type=regressor_type,
                                                                          parameters=self.regressor_parameters)
        return None

    @override
    def create_predictor_factory(self, args, prediction_type: PredictionType) -> SkLearnProblem.PredictorFactory:
        """
        See :func:`mlrl.testbed_sklearn.runnables.SkLearnRunnable.create_predictor_factory`
        """
        return RuleLearnerRunnable.IncrementalPredictionExtension.get_predictor_factory(args, prediction_type)
