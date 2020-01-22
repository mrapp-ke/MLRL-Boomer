#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides a scikit-multilearn implementation of "BOOMER" -- an algorithm for learning gradient boosted multi-label
classification rules. The classifier is composed of several modules, e.g., for rule induction and prediction.
"""
import logging as log
from abc import abstractmethod
from timeit import default_timer as timer

import numpy as np
from boomer.algorithm._head_refinement import HeadRefinement, SingleLabelHeadRefinement, FullHeadRefinement
from boomer.algorithm._losses import Loss, DecomposableLoss, SquaredErrorLoss, LogisticLoss
from boomer.algorithm._pruning import Pruning, IREP
from boomer.algorithm._shrinkage import Shrinkage, ConstantShrinkage
from boomer.algorithm._sub_sampling import FeatureSubSampling, RandomFeatureSubsetSelection
from boomer.algorithm._sub_sampling import InstanceSubSampling, Bagging, RandomInstanceSubsetSelection
from sklearn.utils.validation import check_is_fitted

from boomer.algorithm.model import Theory, DTYPE_FLOAT32
from boomer.algorithm.persistence import ModelPersistence
from boomer.algorithm.prediction import Prediction, Sign, LinearCombination
from boomer.algorithm.rule_induction import RuleInduction, GradientBoosting
from boomer.algorithm.stats import Stats
from boomer.learners import MLLearner


class MLRuleLearner(MLLearner):
    """
    A scikit-multilearn implementation of a rule learner algorithm for multi-label classification or ranking.

    Attributes
        stats_          Statistics about the training data set
        theory_         The theory that contains the classification rules
        persistence     The 'ModelPersistence' to be used to load/save the theory
    """

    STEP_INITIALIZATION = 0

    STEP_RULE_INDUCTION = 1

    PREFIX_RULES = 'rules'

    stats_: Stats

    theory_: Theory

    persistence: ModelPersistence = None

    def __init__(self):
        super().__init__()
        self.require_dense = [True, True]  # We need a dense representation of the training data

    @abstractmethod
    def _create_prediction(self) -> Prediction:
        """
        Must be implemented by subclasses in order to create the `Prediction` to be used for making predictions.

        :return: The `Prediction` that has been created
        """
        pass

    @abstractmethod
    def _create_rule_induction(self) -> RuleInduction:
        """
        Must be implemented by subclasses in order to create the `RuleInduction` to be used for inducing rules.

        :return: The `RuleInduction` that has been created
        """
        pass

    def __load_rules(self):
        """
        Loads the theory from disk, if available.

        :return: The loaded theory, as well as the next step to proceed with
        """
        step = MLRuleLearner.STEP_RULE_INDUCTION

        if self.persistence is not None:
            theory = self.persistence.load_model(model_name=self.get_model_name(),
                                                 file_name_suffix=MLRuleLearner.PREFIX_RULES, fold=self.fold)
        else:
            theory = None

        if theory is None:
            step = MLRuleLearner.STEP_INITIALIZATION

        return theory, step

    def _induce_rules(self, x: np.ndarray, y: np.ndarray, theory: Theory = None) -> Theory:
        """
        Induces classification rules.

        :param x:       An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                        training examples
        :param y:       An array of dtype float, shape `(num_examples, num_labels)`, representing the labels of the
                        training examples
        :param theory:  An existing theory, the induced classification rules should be added to, or None if a new theory
                        should be created
        :return:        A 'Theory' that contains the induced classification rules
        """
        # Create a dense representation of the training data
        x = self._ensure_input_format(x)
        y = self._ensure_input_format(y)

        # Obtain information about the training data
        stats = Stats.create_stats(x, y)
        self.stats_ = stats

        # Load theory from disk, if possible
        model, step = self.__load_rules()

        if model is not None:
            theory = model

        if step == MLRuleLearner.STEP_INITIALIZATION:
            log.info('Inducing classification rules...')
            start_time = timer()

            # Induce rules
            rule_induction = self._create_rule_induction()
            rule_induction.random_state = self.random_state
            theory = rule_induction.induce_rules(stats, x, y, theory)

            # Save theory to disk
            self.__save_rules(theory)

            end_time = timer()
            run_time = end_time - start_time
            num_candidates = len(theory)
            log.info('%s classification rules induced in %s seconds', num_candidates, run_time)

        return theory

    def __save_rules(self, theory: Theory):
        """
        Saves a theory to disk.

        :param theory:  The theory to be saved
        """

        if self.persistence is not None:
            self.persistence.save_model(theory, model_name=self.get_model_name(),
                                        file_name_suffix=MLRuleLearner.PREFIX_RULES, fold=self.fold)

    def fit(self, x: np.ndarray, y: np.ndarray) -> MLLearner:
        self.theory_ = self._induce_rules(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        check_is_fitted(self)

        # Create a dense representation of the given examples
        x = self._ensure_input_format(x)

        # Convert feature matrix into Fortran-contiguous array
        x = np.asfortranarray(x, dtype=DTYPE_FLOAT32)

        log.info("Making a prediction for %s query instances...", np.shape(x)[0])
        prediction = self._create_prediction()
        prediction.random_state = self.random_state
        return prediction.predict(self.stats_, self.theory_, x)

    @abstractmethod
    def get_name(self) -> str:
        pass


class Boomer(MLRuleLearner):
    """
    A scikit-multilearn implementation of "BOOMER" -- an algorithm for learning gradient boosted multi-label
    classification rules.
    """

    def __init__(self, num_rules: int = 100, head_refinement: str = None, loss: str = 'squared-error-loss',
                 instance_sub_sampling: str = None, feature_sub_sampling: str = None, pruning: str = None,
                 shrinkage: float = 1.0):
        """
        :param num_rules:               The number of rules to be induced (including the default rule)
        :param head_refinement:         The strategy that is used to find the heads of rules. Must be `single-label`,
                                        `full` or None, if the default strategy should be used
        :param loss:                    The loss function to be minimized. Must be `squared-error-loss` or
                                        `logistic-loss`
        :param instance_sub_sampling:   The strategy that is used for sub-sampling the training examples each time a new
                                        classification rule is learned. Must be `bagging`, `random-instance-selection`
                                        or None, if no sub-sampling should be used
        :param feature_sub_sampling:    The strategy that is used for sub-sampling the features each time a
                                        classification rule is refined. Must be `random-feature-selection` or None, if
                                        no sub-sampling should be used
        :param pruning:                 The strategy that is used for pruning rules. Must be `irep` or None, if no
                                        pruning should be used
        :param shrinkage:               The shrinkage parameter that should be applied to the predictions of newly
                                        induced rules to reduce their effect on the entire model. Must be in (0, 1]
        """
        super().__init__()
        self.num_rules = num_rules
        self.head_refinement = head_refinement
        self.loss = loss
        self.instance_sub_sampling = instance_sub_sampling
        self.feature_sub_sampling = feature_sub_sampling
        self.pruning = pruning
        self.shrinkage = shrinkage

    def _create_prediction(self) -> Prediction:
        return Sign(LinearCombination())

    def _create_rule_induction(self) -> RuleInduction:
        num_rules = self.num_rules
        loss = self.__create_loss()
        head_refinement = self.__create_head_refinement(loss)
        instance_sub_sampling = self.__create_instance_sub_sampling()
        feature_sub_sampling = self.__create_feature_sub_sampling()
        pruning = self.__create_pruning()
        shrinkage = self.__create_shrinkage()
        return GradientBoosting(num_rules=num_rules, head_refinement=head_refinement, loss=loss,
                                instance_sub_sampling=instance_sub_sampling, feature_sub_sampling=feature_sub_sampling,
                                pruning=pruning, shrinkage=shrinkage)

    def __create_loss(self) -> Loss:
        loss = self.loss

        if loss == 'squared-error-loss':
            return SquaredErrorLoss()
        elif loss == 'logistic-loss':
            return LogisticLoss()
        raise ValueError('Invalid value given for parameter \'loss\': ' + str(loss))

    def __create_head_refinement(self, loss: Loss) -> HeadRefinement:
        head_refinement = self.head_refinement

        if head_refinement is None:
            return SingleLabelHeadRefinement() if isinstance(loss, DecomposableLoss) else FullHeadRefinement()
        elif head_refinement == 'single-label':
            return SingleLabelHeadRefinement()
        elif head_refinement == 'full':
            return FullHeadRefinement()
        raise ValueError('Invalid value given for parameter \'head_refinement\': ' + str(head_refinement))

    def __create_instance_sub_sampling(self) -> InstanceSubSampling:
        instance_sub_sampling = self.instance_sub_sampling

        if instance_sub_sampling is None:
            return None
        elif instance_sub_sampling == 'bagging':
            return Bagging()
        elif instance_sub_sampling == 'random-instance-selection':
            return RandomInstanceSubsetSelection()
        raise ValueError('Invalid value given for parameter \'instance_sub_sampling\': ' + str(instance_sub_sampling))

    def __create_feature_sub_sampling(self) -> FeatureSubSampling:
        feature_sub_sampling = self.feature_sub_sampling

        if feature_sub_sampling is None:
            return None
        elif feature_sub_sampling == 'random-feature-selection':
            return RandomFeatureSubsetSelection()
        raise ValueError('Invalid value given for parameter \'feature_sub_sampling\': ' + str(feature_sub_sampling))

    def __create_pruning(self) -> Pruning:
        pruning = self.pruning

        if pruning is None:
            return None
        if pruning == 'irep':
            return IREP()
        raise ValueError('Invalid value given for parameter \'pruning\': ' + str(pruning))

    def __create_shrinkage(self) -> Shrinkage:
        shrinkage = self.shrinkage

        if 0.0 < shrinkage < 1.0:
            return ConstantShrinkage(shrinkage)
        if shrinkage == 1.0:
            return None
        raise ValueError('Invalid value given for parameter \'shrinkage\': ' + str(shrinkage))

    def get_name(self) -> str:
        num_rules = str(self.num_rules)
        head_refinement = str(self.head_refinement)
        loss = str(self.loss)
        instance_sub_sampling = str(self.instance_sub_sampling)
        feature_sub_sampling = str(self.feature_sub_sampling)
        pruning = str(self.pruning)
        shrinkage = str(self.shrinkage)
        return 'num-rules=' + num_rules + '_head-refinement=' + head_refinement + '_loss=' + loss \
               + '_instance-sub-sampling=' + instance_sub_sampling + '_feature-sub-sampling=' + feature_sub_sampling \
               + '_pruning=' + pruning + '_shrinkage=' + shrinkage
