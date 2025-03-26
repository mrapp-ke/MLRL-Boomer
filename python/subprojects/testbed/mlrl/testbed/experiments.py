"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for performing experiments.
"""
import logging as log

from abc import ABC, abstractmethod
from dataclasses import replace
from functools import reduce
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional

from sklearn.base import BaseEstimator, RegressorMixin, clone

from mlrl.common.mixins import ClassifierMixin, IncrementalClassifierMixin, IncrementalRegressorMixin, \
    NominalFeatureSupportMixin, OrdinalFeatureSupportMixin

from mlrl.testbed.data_splitting import DataSplitter
from mlrl.testbed.dataset import AttributeType, Dataset
from mlrl.testbed.fold import Fold
from mlrl.testbed.format import format_duration
from mlrl.testbed.output.writer import OutputWriter
from mlrl.testbed.output_scope import OutputScope
from mlrl.testbed.parameters import ParameterLoader
from mlrl.testbed.persistence import ModelLoader, ModelSaver
from mlrl.testbed.prediction_result import PredictionResult
from mlrl.testbed.prediction_scope import GlobalPrediction, IncrementalPrediction, PredictionType
from mlrl.testbed.problem_type import ProblemType
from mlrl.testbed.training_result import TrainingResult


class Evaluation(ABC):
    """
    An abstract base class for all classes that allow to evaluate predictions that are obtained from a previously
    trained model.
    """

    def __init__(self, prediction_type: PredictionType, output_writers: List[OutputWriter]):
        """
        :param prediction_type: The type of the predictions to be obtained
        :param output_writers:  A list that contains all output writers to be invoked after predictions have been
                                obtained
        """
        self.prediction_type = prediction_type
        self.output_writers = output_writers

    def _invoke_prediction_function(self, learner, predict_function, predict_proba_function, dataset: Dataset,
                                    **kwargs):
        """
        May be used by subclasses in order to invoke the correct prediction function, depending on the type of
        result that should be obtained.

        :param learner:                 The learner, the result should be obtained from
        :param predict_function:        The function to be invoked if binary results or scores should be obtained
        :param predict_proba_function:  The function to be invoked if probability estimates should be obtained
        :param dataset:                 The dataset that stores the query examples
        :param kwargs:                  Optional keyword arguments to be passed to the `predict_function`
        :return:                        The return value of the invoked function
        """
        prediction_type = self.prediction_type
        x = dataset.x

        if prediction_type == PredictionType.SCORES:
            try:
                if isinstance(learner, ClassifierMixin):
                    result = predict_function(x, predict_scores=True, **kwargs)
                elif isinstance(learner, RegressorMixin):
                    result = predict_function(x, **kwargs)
                else:
                    raise RuntimeError()
            except RuntimeError:
                log.error('Prediction of scores not supported')
                result = None
        elif prediction_type == PredictionType.PROBABILITIES:
            try:
                result = predict_proba_function(x)
            except RuntimeError:
                log.error('Prediction of probabilities not supported')
                result = None
        else:
            result = predict_function(x, **kwargs)

        return result

    def _evaluate_predictions(self, scope: OutputScope, prediction_result: PredictionResult, train_time: float,
                              learner):
        """
        May be used by subclasses in order to evaluate predictions that have been obtained from a previously trained
        model.

        :param scope:               The scope of the output data
        :param prediction_result:   A `PredictionResult` that provides access to the predictions have been obtained
        :param train_time:          The time needed to train the model
        :param learner:             The learner, the predictions have been obtained from
        """
        for output_writer in self.output_writers:
            output_writer.write_output(scope, TrainingResult(learner=learner, train_time=train_time), prediction_result)

    @abstractmethod
    def predict_and_evaluate(self, scope: OutputScope, training_result: TrainingResult, **kwargs):
        """
        Must be implemented by subclasses in order to obtain and evaluate predictions for given query examples from a
        previously trained model.

        :param scope:           The scope of the output data
        :param training_result: A `TrainingResult` that stores the result of the training process
        :param kwargs:          Optional keyword arguments to be passed to the model when obtaining predictions
        """


class GlobalEvaluation(Evaluation):
    """
    Obtains and evaluates predictions from a previously trained global model.
    """

    def predict_and_evaluate(self, scope: OutputScope, training_result: TrainingResult, **kwargs):
        dataset = scope.dataset
        log.info('Predicting for %s %s examples...', dataset.num_examples, dataset.type.value)
        learner = training_result.learner
        start_time = timer()
        predict_proba_function = learner.predict_proba if callable(getattr(learner, 'predict_proba', None)) else None
        predictions = self._invoke_prediction_function(learner, learner.predict, predict_proba_function, dataset,
                                                       **kwargs)
        end_time = timer()
        predict_time = end_time - start_time

        if predictions is not None:
            log.info('Successfully predicted in %s', format_duration(predict_time))
            prediction_result = PredictionResult(predictions=predictions,
                                                 prediction_type=self.prediction_type,
                                                 prediction_scope=GlobalPrediction(),
                                                 predict_time=predict_time)
            self._evaluate_predictions(scope=scope,
                                       prediction_result=prediction_result,
                                       train_time=training_result.train_time,
                                       learner=learner)


class IncrementalEvaluation(Evaluation):
    """
    Repeatedly obtains and evaluates predictions from a previously trained ensemble model, e.g., a model consisting of
    several rules, using only a subset of the ensemble members with increasing size.
    """

    def __init__(self, prediction_type: PredictionType, output_writers: List[OutputWriter], min_size: int,
                 max_size: int, step_size: int):
        """
        :param min_size:    The minimum number of ensemble members to be evaluated. Must be at least 0
        :param max_size:    The maximum number of ensemble members to be evaluated. Must be greater than `min_size` or
                            0, if all ensemble members should be evaluated
        :param step_size:   The number of additional ensemble members to be considered at each repetition. Must be at
                            least 1
        """
        super().__init__(prediction_type, output_writers)
        self.min_size = min_size
        self.max_size = max_size
        self.step_size = step_size

    def predict_and_evaluate(self, scope: OutputScope, training_result: TrainingResult, **kwargs):
        learner = training_result.learner

        if not isinstance(learner, IncrementalClassifierMixin) and not isinstance(learner, IncrementalRegressorMixin):
            raise ValueError('Cannot obtain incremental predictions from a model of type ' + type(learner.__name__))

        predict_proba_function = learner.predict_proba_incrementally if callable(
            getattr(learner, 'predict_proba_incrementally', None)) else None
        dataset = scope.dataset
        incremental_predictor = self._invoke_prediction_function(learner, learner.predict_incrementally,
                                                                 predict_proba_function, dataset, **kwargs)

        if incremental_predictor:
            step_size = self.step_size
            total_size = incremental_predictor.get_num_next()
            max_size = self.max_size

            if max_size > 0:
                total_size = min(max_size, total_size)

            min_size = self.min_size
            next_step_size = min_size if min_size > 0 else step_size
            current_size = min(next_step_size, total_size)

            while incremental_predictor.has_next():
                log.info('Predicting for %s %s examples using a model of size %s...', dataset.num_examples,
                         dataset.type.value, current_size)
                start_time = timer()
                predictions = incremental_predictor.apply_next(next_step_size)
                end_time = timer()
                predict_time = end_time - start_time

                if predictions is not None:
                    log.info('Successfully predicted in %s', format_duration(predict_time))
                    prediction_result = PredictionResult(predictions=predictions,
                                                         prediction_type=self.prediction_type,
                                                         prediction_scope=IncrementalPrediction(current_size),
                                                         predict_time=predict_time)
                    self._evaluate_predictions(scope=scope,
                                               prediction_result=prediction_result,
                                               train_time=training_result.train_time,
                                               learner=learner)

                next_step_size = step_size
                current_size = min(current_size + next_step_size, total_size)


class Experiment(DataSplitter.Callback):
    """
    An experiment that trains and evaluates a machine learning model on a specific data set using cross validation or
    separate training and test sets.
    """

    class ExecutionHook(ABC):
        """
        An abstract base class for all operations that may be executed before or after an experiment.
        """

        @abstractmethod
        def execute(self):
            """
            Must be overridden by subclasses in order to execute the operation.
            """

    def __init__(self,
                 problem_type: ProblemType,
                 base_learner: BaseEstimator,
                 learner_name: str,
                 data_splitter: DataSplitter,
                 pre_training_output_writers: List[OutputWriter],
                 post_training_output_writers: List[OutputWriter],
                 pre_execution_hook: Optional[ExecutionHook] = None,
                 train_evaluation: Optional[Evaluation] = None,
                 test_evaluation: Optional[Evaluation] = None,
                 parameter_loader: Optional[ParameterLoader] = None,
                 model_loader: Optional[ModelLoader] = None,
                 model_saver: Optional[ModelSaver] = None,
                 fit_kwargs: Optional[Dict[str, Any]] = None,
                 predict_kwargs: Optional[Dict[str, Any]] = None):
        """
        :param problem_type:                    The type of the machine learning problem
        :param base_learner:                    The machine learning algorithm to be used
        :param learner_name:                    The name of the machine learning algorithm
        :param data_splitter:                   The method to be used for splitting the available data into training and
                                                test sets
        :param pre_training_output_writers:     A list that contains all output writers to be invoked before training
        :param post_training_output_writers:    A list that contains all output writers to be invoked after training
        :param pre_execution_hook:              An operation that should be executed before the experiment
        :param train_evaluation:                The method to be used for evaluating the predictions for the training
                                                data or None, if the predictions should not be evaluated
        :param test_evaluation:                 The method to be used for evaluating the predictions for the test data
                                                or None, if the predictions should not be evaluated
        :param parameter_loader:                The `ParameterLoader` that should be used to read the parameter settings
        :param model_loader:                    The `ModelLoader` that should be used for loading models
        :param model_saver:                     The `ModelSaver` that should be used for saving models
        :param fit_kwargs:                      Optional keyword arguments to be passed to the learner when fitting a
                                                model
        :param predict_kwargs:                  Optional keyword arguments to be passed to the learner when obtaining
                                                predictions from a model
        """
        self.problem_type = problem_type
        self.base_learner = base_learner
        self.learner_name = learner_name
        self.data_splitter = data_splitter
        self.pre_training_output_writers = pre_training_output_writers
        self.post_training_output_writers = post_training_output_writers
        self.pre_execution_hook = pre_execution_hook
        self.train_evaluation = train_evaluation
        self.test_evaluation = test_evaluation
        self.parameter_loader = parameter_loader
        self.model_loader = model_loader
        self.model_saver = model_saver
        self.fit_kwargs = fit_kwargs
        self.predict_kwargs = predict_kwargs

    def run(self):
        """
        Runs the experiment.
        """
        log.info('Starting experiment using the %s algorithm "%s"...', self.problem_type.value, self.learner_name)

        # Run pre-execution hook, if necessary...
        pre_execution_hook = self.pre_execution_hook

        if pre_execution_hook:
            pre_execution_hook.execute()

        self.data_splitter.run(self)

    def train_and_evaluate(self, fold: Fold, train_dataset: Dataset, test_dataset: Dataset):
        """
        See `DataSplitter.Callback.train_and_evaluate`
        """
        problem_type = self.problem_type
        learner = clone(self.base_learner)
        fit_kwargs = self.fit_kwargs if self.fit_kwargs else {}

        # Apply parameter setting, if necessary...
        parameter_loader = self.parameter_loader

        if parameter_loader:
            parameters = parameter_loader.load_parameters(fold)

            if parameters:
                learner.set_params(**parameters)
                log.info('Successfully applied parameter setting: %s', parameters)
        else:
            parameters = learner.get_params()

        # Write output data before model is trained...
        train_scope = OutputScope(problem_type=problem_type, dataset=train_dataset, fold=fold, parameters=parameters)

        for output_writer in self.pre_training_output_writers:
            output_writer.write_output(train_scope)

        # Set the indices of ordinal features, if supported...
        if isinstance(learner, OrdinalFeatureSupportMixin):
            fit_kwargs[OrdinalFeatureSupportMixin.KWARG_ORDINAL_FEATURE_INDICES] = train_dataset.get_feature_indices(
                AttributeType.ORDINAL)

        # Set the indices of nominal features, if supported...
        if isinstance(learner, NominalFeatureSupportMixin):
            fit_kwargs[NominalFeatureSupportMixin.KWARG_NOMINAL_FEATURE_INDICES] = train_dataset.get_feature_indices(
                AttributeType.NOMINAL)

        # Load model from disk, if possible, otherwise train a new model...
        loaded_learner = self.__load_model(fold)

        if isinstance(loaded_learner, type(learner)):
            self.__check_for_parameter_changes(expected_params=parameters, actual_params=loaded_learner.get_params())
            loaded_learner.set_params(**parameters)
            learner = loaded_learner
            train_time = 0
        else:
            log.info('Fitting model to %s training examples...', train_dataset.num_examples)
            train_time = self.__train(learner, train_dataset, **fit_kwargs)
            log.info('Successfully fit model in %s', format_duration(train_time))

            # Save model to disk...
            self.__save_model(learner, fold)

        training_result = TrainingResult(learner=learner, train_time=train_time)

        # Obtain and evaluate predictions for training data, if necessary...
        train_evaluation = self.train_evaluation

        if train_evaluation and test_dataset.type != Dataset.Type.TRAINING:
            predict_kwargs = self.predict_kwargs if self.predict_kwargs else {}
            self.__predict_and_evaluate(train_scope, training_result, train_evaluation, **predict_kwargs)

        # Obtain and evaluate predictions for test data, if necessary...
        test_evaluation = self.test_evaluation

        if test_evaluation:
            test_scope = OutputScope(problem_type=problem_type, dataset=test_dataset, fold=fold, parameters=parameters)
            predict_kwargs = self.predict_kwargs if self.predict_kwargs else {}
            self.__predict_and_evaluate(test_scope, training_result, test_evaluation, **predict_kwargs)

        # Write output data after model was trained...
        for output_writer in self.post_training_output_writers:
            output_writer.write_output(train_scope, training_result=training_result)

    @staticmethod
    def __predict_and_evaluate(scope: OutputScope, training_result: TrainingResult, evaluation: Evaluation, **kwargs):
        """
        Obtains and evaluates predictions for given query examples from a previously trained model.

        :param scope:           The scope of the output data
        :param training_result: A `TrainingResult` that stores the result of the training process
        :param evaluation:      The `Evaluation` to be used
        :param kwargs:          Optional keyword arguments to be passed to the model when obtaining predictions
        """
        try:
            return evaluation.predict_and_evaluate(scope, training_result, **kwargs)
        except ValueError as error:
            dataset = scope.dataset

            if dataset.has_sparse_features:
                dense_dataset = replace(scope, dataset=dataset.enforce_dense_features())
                return Experiment.__predict_and_evaluate(dense_dataset, evaluation, training_result, **kwargs)

            raise error

    @staticmethod
    def __train(learner, dataset: Dataset, **kwargs):
        """
        Fits a learner to training data.

        :param learner: The learner
        :param dataset: The training dataset
        :param kwargs:  Optional keyword arguments to be passed to the learner when fitting model
        :return:        The time needed for training
        """
        try:
            start_time = timer()
            learner.fit(dataset.x, dataset.y, **kwargs)
            end_time = timer()
            return end_time - start_time
        except ValueError as error:
            if dataset.has_sparse_features:
                return Experiment.__train(learner, dataset.enforce_dense_features(), **kwargs)
            if dataset.has_sparse_outputs:
                return Experiment.__train(learner, dataset.enforce_dense_outputs(), **kwargs)
            raise error

    def __load_model(self, fold: Fold):
        """
        Loads the model from disk, if available.

        :param fold:    The fold of the available data, the model corresponds to
        :return:        The loaded model
        """
        model_loader = self.model_loader

        if model_loader:
            return model_loader.load_model(self.learner_name, fold)

        return None

    def __save_model(self, model, fold: Fold):
        """
        Saves a model to disk.

        :param model:   The model to be saved
        :param fold:    The fold of the available data, the model corresponds to
        """
        model_saver = self.model_saver

        if model_saver:
            model_saver.save_model(model, self.learner_name, fold)

    @staticmethod
    def __check_for_parameter_changes(expected_params, actual_params):
        changes = []

        for key, expected_value in expected_params.items():
            expected_value = str(expected_value)
            actual_value = str(actual_params[key])

            if actual_value != expected_value:
                changes.append((key, expected_value, actual_value))

        if changes:
            log.warning(
                'The loaded model\'s values for the following parameters differ from the expected configuration: %s',
                reduce(
                    lambda aggr, change: aggr +
                    (', '
                     if aggr else '') + '"' + change[0] + '" is "' + change[2] + '" instead of "' + change[1] + '"',
                    changes, ''))
