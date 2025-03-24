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

from sklearn.base import BaseEstimator as SkLearnBaseEstimator, RegressorMixin as SkLearnRegressorMixin, clone

from mlrl.common.data.arrays import is_sparse
from mlrl.common.mixins import ClassifierMixin, IncrementalClassifierMixin, IncrementalRegressorMixin, \
    NominalFeatureSupportMixin, OrdinalFeatureSupportMixin

from mlrl.testbed.data_splitting import DataSplitter
from mlrl.testbed.dataset import AttributeType, Dataset
from mlrl.testbed.fold import Fold
from mlrl.testbed.format import format_duration
from mlrl.testbed.output_writer import OutputWriter
from mlrl.testbed.parameters import ParameterLoader
from mlrl.testbed.persistence import ModelLoader, ModelSaver
from mlrl.testbed.prediction_scope import GlobalPrediction, IncrementalPrediction, PredictionScope, PredictionType
from mlrl.testbed.problem_type import ProblemType


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
                elif isinstance(learner, SkLearnRegressorMixin):
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

    def _evaluate_predictions(self, problem_type: ProblemType, fold: Fold, data_type: Dataset.Type,
                              prediction_scope: PredictionScope, train_time: float, predict_time: float,
                              dataset: Dataset, predictions, learner):
        """
        May be used by subclasses in order to evaluate predictions that have been obtained from a previously trained
        model.

        :param problem_type:        The type of the machine learning problem
        :param fold:                The fold of the available data, the predictions and ground truth correspond to
        :param data_type:           Specifies whether the predictions and ground truth correspond to the training or
                                    test data
        :param prediction_scope:    Specifies whether the predictions have been obtained from a global model or
                                    incrementally
        :param train_time:          The time needed to train the model
        :param predict_time:        The time needed to obtain the predictions
        :param dataset:             The dataset for which the predictions have been obtained
        :param predictions:         A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray` matrix, shape
                                    `(num_examples, num_outputs)`, that stores the predictions for the query examples
        :param learner:             The learner, the predictions have been obtained from
        """
        for output_writer in self.output_writers:
            output_writer.write_output(problem_type, dataset, fold, learner, data_type, self.prediction_type,
                                       prediction_scope, predictions, train_time, predict_time)

    @abstractmethod
    def predict_and_evaluate(self, problem_type: ProblemType, fold: Fold, data_type: Dataset.Type, train_time: float,
                             learner, test_dataset: Dataset, **kwargs):
        """
        Must be implemented by subclasses in order to obtain and evaluate predictions for given query examples from a
        previously trained model.

        :param problem_type:    The type of the machine learning problem
        :param fold:            The fold of the available data, the predictions and ground truth correspond to
        :param data_type:       Specifies whether the predictions and ground truth correspond to the training or test
                                data
        :param train_time:      The time needed to train the model
        :param learner:         The learner, the predictions should be obtained from
        :param test_dataset:    The dataset that stores the query examples
        :param kwargs:          Optional keyword arguments to be passed to the model when obtaining predictions
        """


class GlobalEvaluation(Evaluation):
    """
    Obtains and evaluates predictions from a previously trained global model.
    """

    def predict_and_evaluate(self, problem_type: ProblemType, fold: Fold, data_type: Dataset.Type, train_time: float,
                             learner, test_dataset: Dataset, **kwargs):
        log.info('Predicting for %s %s examples...', test_dataset.num_examples, data_type.value)
        start_time = timer()
        predict_proba_function = learner.predict_proba if callable(getattr(learner, 'predict_proba', None)) else None
        predictions = self._invoke_prediction_function(learner, learner.predict, predict_proba_function, test_dataset,
                                                       **kwargs)
        end_time = timer()
        predict_time = end_time - start_time

        if predictions is not None:
            log.info('Successfully predicted in %s', format_duration(predict_time))
            self._evaluate_predictions(problem_type=problem_type,
                                       fold=fold,
                                       data_type=data_type,
                                       prediction_scope=GlobalPrediction(),
                                       train_time=train_time,
                                       predict_time=predict_time,
                                       dataset=test_dataset,
                                       predictions=predictions,
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

    def predict_and_evaluate(self, problem_type: ProblemType, fold: Fold, data_type: Dataset.Type, train_time: float,
                             learner, test_dataset: Dataset, **kwargs):
        if not isinstance(learner, IncrementalClassifierMixin) and not isinstance(learner, IncrementalRegressorMixin):
            raise ValueError('Cannot obtain incremental predictions from a model of type ' + type(learner.__name__))

        predict_proba_function = learner.predict_proba_incrementally if callable(
            getattr(learner, 'predict_proba_incrementally', None)) else None
        incremental_predictor = self._invoke_prediction_function(learner, learner.predict_incrementally,
                                                                 predict_proba_function, test_dataset, **kwargs)

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
                log.info('Predicting for %s %s examples using a model of size %s...', test_dataset.num_examples,
                         data_type.value, current_size)
                start_time = timer()
                predictions = incremental_predictor.apply_next(next_step_size)
                end_time = timer()
                predict_time = end_time - start_time

                if predictions is not None:
                    log.info('Successfully predicted in %s', format_duration(predict_time))
                    self._evaluate_predictions(problem_type=problem_type,
                                               fold=fold,
                                               data_type=data_type,
                                               prediction_scope=IncrementalPrediction(current_size),
                                               train_time=train_time,
                                               predict_time=predict_time,
                                               dataset=test_dataset,
                                               predictions=predictions,
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
                 base_learner: SkLearnBaseEstimator,
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
        base_learner = self.base_learner
        current_learner = clone(base_learner)
        fit_kwargs = self.fit_kwargs if self.fit_kwargs else {}

        # Apply parameter setting, if necessary...
        parameter_loader = self.parameter_loader

        if parameter_loader:
            params = parameter_loader.load_parameters(fold)

            if params:
                current_learner.set_params(**params)
                log.info('Successfully applied parameter setting: %s', params)

        # Write output data before model is trained...
        for output_writer in self.pre_training_output_writers:
            output_writer.write_output(problem_type, train_dataset, fold, current_learner)

        # Set the indices of ordinal features, if supported...
        if isinstance(current_learner, OrdinalFeatureSupportMixin):
            meta_data = train_dataset.meta_data
            fit_kwargs[OrdinalFeatureSupportMixin.KWARG_ORDINAL_FEATURE_INDICES] = meta_data.get_feature_indices(
                AttributeType.ORDINAL)

        # Set the indices of nominal features, if supported...
        if isinstance(current_learner, NominalFeatureSupportMixin):
            meta_data = train_dataset.meta_data
            fit_kwargs[NominalFeatureSupportMixin.KWARG_NOMINAL_FEATURE_INDICES] = meta_data.get_feature_indices(
                AttributeType.NOMINAL)

        # Load model from disk, if possible, otherwise train a new model...
        loaded_learner = self.__load_model(fold)

        if isinstance(loaded_learner, type(current_learner)):
            current_params = current_learner.get_params()
            self.__check_for_parameter_changes(expected_params=current_params,
                                               actual_params=loaded_learner.get_params())
            loaded_learner.set_params(**current_params)
            current_learner = loaded_learner
            train_time = 0
        else:
            log.info('Fitting model to %s training examples...', train_dataset.num_examples)
            train_time = self.__train(current_learner, train_dataset, **fit_kwargs)
            log.info('Successfully fit model in %s', format_duration(train_time))

            # Save model to disk...
            self.__save_model(current_learner, fold)

        # Obtain and evaluate predictions for training data, if necessary...
        evaluation = self.train_evaluation

        if evaluation and fold.is_train_test_separated:
            data_type = Dataset.Type.TRAINING
            predict_kwargs = self.predict_kwargs if self.predict_kwargs else {}
            self.__predict_and_evaluate(problem_type, evaluation, fold, data_type, train_time, current_learner,
                                        train_dataset, **predict_kwargs)

        # Obtain and evaluate predictions for test data, if necessary...
        evaluation = self.test_evaluation

        if evaluation:
            data_type = Dataset.Type.TEST if fold.is_train_test_separated else Dataset.Type.TRAINING
            predict_kwargs = self.predict_kwargs if self.predict_kwargs else {}
            self.__predict_and_evaluate(problem_type, evaluation, fold, data_type, train_time, current_learner,
                                        test_dataset, **predict_kwargs)

        # Write output data after model was trained...
        for output_writer in self.post_training_output_writers:
            output_writer.write_output(problem_type, train_dataset, fold, current_learner, train_time=train_time)

    @staticmethod
    def __predict_and_evaluate(problem_type: ProblemType, evaluation: Evaluation, fold: Fold, data_type: Dataset.Type,
                               train_time: float, learner, test_dataset: Dataset, **kwargs):
        """
        Obtains and evaluates predictions for given query examples from a previously trained model.

        :param problem_type:    The type of the machine learning problem
        :param evaluation:      The `Evaluation` to be used
        :param fold:            The fold of the available data, the predictions and ground truth correspond to
        :param data_type:       Specifies whether the predictions and ground truth correspond to the training or test
                                data
        :param train_time:      The time needed to train the model
        :param learner:         The learner, the predictions should be obtained from
        :param test_dataset:    The dataset that stores the query examples
        :param kwargs:          Optional keyword arguments to be passed to the model when obtaining predictions
        """
        try:
            return evaluation.predict_and_evaluate(problem_type, fold, data_type, train_time, learner, test_dataset,
                                                   **kwargs)
        except ValueError as error:
            if is_sparse(test_dataset.x):
                return Experiment.__predict_and_evaluate(problem_type, evaluation, fold, data_type, train_time, learner,
                                                         replace(test_dataset, x=test_dataset.x.toarray()), **kwargs)
            raise error

    @staticmethod
    def __train(learner, train_dataset: Dataset, **kwargs):
        """
        Fits a learner to training data.

        :param learner: The learner
        :param train_dataset: The training dataset
        :param kwargs:  Optional keyword arguments to be passed to the learner when fitting model
        :return:        The time needed for training
        """
        x = train_dataset.x
        y = train_dataset.y

        try:
            start_time = timer()
            learner.fit(x, y, **kwargs)
            end_time = timer()
            return end_time - start_time
        except ValueError as error:
            if is_sparse(y):
                return Experiment.__train(learner, replace(train_dataset, y=y.toarray()), **kwargs)
            if is_sparse(x):
                return Experiment.__train(learner, replace(train_dataset, x=x.toarray()), **kwargs)
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
