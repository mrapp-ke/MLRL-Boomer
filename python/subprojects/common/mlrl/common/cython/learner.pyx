"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython.validation import assert_greater_or_equal
from mlrl.common.cython.feature_info cimport FeatureInfo
from mlrl.common.cython.feature_matrix cimport ColumnWiseFeatureMatrix, RowWiseFeatureMatrix
from mlrl.common.cython.label_matrix cimport RowWiseLabelMatrix
from mlrl.common.cython.label_space_info cimport create_label_space_info
from mlrl.common.cython.prediction cimport BinaryPredictor, SparseBinaryPredictor, ScorePredictor, ProbabilityPredictor
from mlrl.common.cython.rule_induction cimport GreedyTopDownRuleInductionConfig
from mlrl.common.cython.rule_model cimport create_rule_model

from libcpp.utility cimport move

from cython.operator cimport dereference


cdef class TrainingResult:
    """
    Provides access to the results of fitting a rule learner to training data. It incorporates the model that has been
    trained, as well as additional information that is necessary for obtaining predictions for unseen data.
    """

    def __cinit__(self, uint32 num_labels, RuleModel rule_model not None, LabelSpaceInfo label_space_info not None):
        """
        :param num_labels:          The number of labels for which a model has been trained
        :param rule_model:          The `RuleModel` that has been trained
        :param label_space_info:    The `LabelSpaceInfo` that may be used as a basis for making predictions
        """
        self.num_labels = num_labels
        self.rule_model = rule_model
        self.label_space_info = label_space_info


cdef class RuleLearnerConfig:
    """
    Allows to configure a rule learner.
    """

    cdef IRuleLearnerConfig* get_rule_learner_config_ptr(self):
        pass

    def use_sequential_rule_model_assemblage(self):
        """
        Configures the rule learner to use an algorithm that sequentially induces several rules, optionally starting
        with a default rule, that are added to a rule-based model.
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        rule_learner_config_ptr.useSequentialRuleModelAssemblage()

    def use_greedy_top_down_rule_induction(self) -> GreedyTopDownRuleInductionConfig:
        """
        Configures the algorithm to use a greedy top-down search for the induction of individual rules.

        :return: A `GreedyTopDownRuleInductionConfig` that allows further configuration of the algorithm for the
                 induction of individual rules
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        cdef IGreedyTopDownRuleInductionConfig* config_ptr = &rule_learner_config_ptr.useGreedyTopDownRuleInduction()
        cdef GreedyTopDownRuleInductionConfig config = GreedyTopDownRuleInductionConfig.__new__(GreedyTopDownRuleInductionConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_rule_pruning(self):
        """
        Configures the rule learner to not prune individual rules.
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        rule_learner_config_ptr.useNoRulePruning()

    def use_no_post_processor(self):
        """
        Configures the rule learner to not use any post-processor.
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        rule_learner_config_ptr.useNoPostProcessor()

    def use_no_parallel_rule_refinement(self):
        """
        Configures the rule learner to not use any multi-threading for the parallel refinement of rules.
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        rule_learner_config_ptr.useNoParallelRuleRefinement()

    def use_no_parallel_statistic_update(self):
        """
        Configures the rule learner to not use any multi-threading for the parallel update of statistics.
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        rule_learner_config_ptr.useNoParallelStatisticUpdate()

    def use_no_parallel_prediction(self):
        """
        Configures the rule learner to not use any multi-threading to predict for several query examples in parallel.
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        rule_learner_config_ptr.useNoParallelPrediction()

    def use_no_size_stopping_criterion(self):
        """
        Configures the rule learner to not use a stopping criterion that ensures that the number of induced rules does
        not exceed a certain maximum.
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        rule_learner_config_ptr.useNoSizeStoppingCriterion()

    def use_no_time_stopping_criterion(self):
        """
        Configures the rule learner to not use a stopping criterion that ensures that are certain time limit is not
        exceeded.
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        rule_learner_config_ptr.useNoTimeStoppingCriterion()

    def use_no_global_pruning(self):
        """
        Configures the rule learner to not use a stopping criterion that allows to decide how many rules should be
        included in a model, such that its performance is optimized globally.
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        rule_learner_config_ptr.useNoGlobalPruning()

    def use_no_sequential_post_optimization(self):
        """
        Configures the rule learner to not use a post-optimization method that optimizes each rule in a model by
        relearning it in the context of the other rules.
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        rule_learner_config_ptr.useNoSequentialPostOptimization()


cdef class RuleLearner:
    """
    A rule learner.
    """

    cdef IRuleLearner* get_rule_learner_ptr(self):
        pass

    def fit(self, FeatureInfo feature_info not None, ColumnWiseFeatureMatrix feature_matrix not None,
            RowWiseLabelMatrix label_matrix not None, uint32 random_state) -> TrainingResult:
        """
        Applies the rule learner to given training examples and corresponding ground truth labels.

        :param feature_info:    A `FeatureInfo` that provides information about the types of individual features
        :param feature_matrix:  A `ColumnWiseFeatureMatrix` that provides column-wise access to the feature values of
                                the training examples
        :param label_matrix:    A `RowWiseLabelMatrix` that provides row-wise access to the ground truth labels of the
                                training examples
        :param random_state:    The seed to be used by random number generators
        :return:                The `TrainingResult` that provides access to the result of fitting the rule learner to
                                the training data
        """
        assert_greater_or_equal("random_state", random_state, 1)
        cdef unique_ptr[ITrainingResult] training_result_ptr = self.get_rule_learner_ptr().fit(
            dereference(feature_info.get_feature_info_ptr()),
            dereference(feature_matrix.get_column_wise_feature_matrix_ptr()),
            dereference(label_matrix.get_row_wise_label_matrix_ptr()), random_state)
        cdef uint32 num_labels = training_result_ptr.get().getNumLabels()
        cdef unique_ptr[IRuleModel] rule_model_ptr = move(training_result_ptr.get().getRuleModel())
        cdef unique_ptr[ILabelSpaceInfo] label_space_info_ptr = move(training_result_ptr.get().getLabelSpaceInfo())
        cdef RuleModel rule_model = create_rule_model(move(rule_model_ptr))
        cdef LabelSpaceInfo label_space_info = create_label_space_info(move(label_space_info_ptr))
        return TrainingResult.__new__(TrainingResult, num_labels, rule_model, label_space_info)

    def can_predict_binary(self, RowWiseFeatureMatrix feature_matrix not None, uint32 num_labels) -> bool:
        """
        Returns whether the rule learner is able to predict binary labels or not.

        :param feature_matrix:  A `RowWiseFeatureMatrix` that provides row-wise access to the feature values of the
                                query examples
        :param num_labels:      The number of labels to predict for
        :return:                True, if the rule learner is able to predict binary labels, False otherwise
        """
        return self.get_rule_learner_ptr().canPredictBinary(
            dereference(feature_matrix.get_row_wise_feature_matrix_ptr()), num_labels)

    def create_binary_predictor(self, RowWiseFeatureMatrix feature_matrix not None, RuleModel rule_model not None,
                                LabelSpaceInfo label_space_info not None, uint32 num_labels) -> BinaryPredictor:
        """
        Creates and returns a predictor that may be used to predict binary labels for given query examples. If the
        prediction of binary labels is not supported by the rule learner, a `RuntimeError` is thrown.

        :param feature_matrix:      A `RowWiseFeatureMatrix` that provides row-wise access to the feature values of the
                                    query examples
        :param rule_model:          The `RuleModel` that should be used to obtain predictions
        :param label_space_info:    The `LabelSpaceInfo` that provides information about the label space that may be
                                    used as a basis for obtaining predictions
        :param num_labels:          The number of labels to predict for
        :return:                    A `BinaryPredictor` that may be used to predict binary labels for the given query
                                    examples
        """
        cdef unique_ptr[IBinaryPredictor] predictor_ptr = move(self.get_rule_learner_ptr().createBinaryPredictor(
            dereference(feature_matrix.get_row_wise_feature_matrix_ptr()),
            dereference(rule_model.get_rule_model_ptr()),
            dereference(label_space_info.get_label_space_info_ptr()),
            num_labels))
        cdef BinaryPredictor binary_predictor = BinaryPredictor.__new__(BinaryPredictor)
        binary_predictor.predictor_ptr = move(predictor_ptr)
        return binary_predictor

    def create_sparse_binary_predictor(self, RowWiseFeatureMatrix feature_matrix not None,
                                       RuleModel rule_model not None, LabelSpaceInfo label_space_info not None,
                                       uint32 num_labels) -> SparseBinaryPredictor:
        """
        Creates and returns a predictor that may be used to predict sparse binary labels for given query examples. If
        the prediction of sparse binary labels is not supported by the rule learner, a `RuntimeError` is thrown.

        :param feature_matrix:      A `RowWiseFeatureMatrix` that provides row-wise access to the feature values of the
                                    query examples
        :param rule_model:          The `RuleModel` that should be used to obtain predictions
        :param label_space_info:    The `LabelSpaceInfo` that provides information about the label space that may be
                                    used as a basis for obtaining predictions
        :param num_labels:          The number of labels to predict for
        :return:                    A `SparseBinaryPredictor` that may be used to predict sparse binary labels for the
                                    given query examples
        """
        cdef unique_ptr[ISparseBinaryPredictor] predictor_ptr = move(self.get_rule_learner_ptr().createSparseBinaryPredictor(
            dereference(feature_matrix.get_row_wise_feature_matrix_ptr()),
            dereference(rule_model.get_rule_model_ptr()),
            dereference(label_space_info.get_label_space_info_ptr()),
            num_labels))
        cdef SparseBinaryPredictor sparse_binary_predictor = SparseBinaryPredictor.__new__(SparseBinaryPredictor)
        sparse_binary_predictor.predictor_ptr = move(predictor_ptr)
        return sparse_binary_predictor

    def can_predict_scores(self, RowWiseFeatureMatrix feature_matrix not None, uint32 num_labels) -> bool:
        """
        Returns whether the rule learner is able to predict regression scores or not.

        :param feature_matrix:  A `RowWiseFeatureMatrix` that provides row-wise access to the feature values of the
                                query examples
        :param num_labels:      The number of labels to predict for
        return:                 True, if the rule learner is able to predict regression scores, False otherwise
        """
        return self.get_rule_learner_ptr().canPredictScores(
            dereference(feature_matrix.get_row_wise_feature_matrix_ptr()), num_labels)

    def create_score_predictor(self, RowWiseFeatureMatrix feature_matrix not None, RuleModel rule_model not None,
                               LabelSpaceInfo label_space_info not None, uint32 num_labels) -> ScorePredictor:
        """
        Creates and returns a predictor that may be used to predict regression scores for given query examples. If the
        prediction of regression scores is not supported by the rule learner, a `RuntimeError` is thrown.

        :param feature_matrix:      A `RowWiseFeatureMatrix` that provides row-wise access to the feature values of the
                                    query examples
        :param rule_model:          The `RuleModel` that should be used to obtain predictions
        :param label_space_info:    The `LabelSpaceInfo` that provides information about the label space that may be
                                    used as a basis for obtaining predictions
        :param num_labels:          The number of labels to predict for
        :return:                    A `ScorePredictor` that may be used to predict regression scores for the given query
                                    examples
        """
        cdef unique_ptr[IScorePredictor] predictor_ptr = move(self.get_rule_learner_ptr().createScorePredictor(
            dereference(feature_matrix.get_row_wise_feature_matrix_ptr()),
            dereference(rule_model.get_rule_model_ptr()),
            dereference(label_space_info.get_label_space_info_ptr()),
            num_labels))
        cdef ScorePredictor score_predictor = ScorePredictor.__new__(ScorePredictor)
        score_predictor.predictor_ptr = move(predictor_ptr)
        return score_predictor

    def can_predict_probabilities(self, RowWiseFeatureMatrix feature_matrix not None, uint32 num_labels) -> bool:
        """
        Returns whether the rule learner is able to predict probability estimates or not.

        :param feature_matrix:  A `RowWiseFeatureMatrix` that provides row-wise access to the feature values of the
                                query examples
        :param num_labels:      The number of labels to predict for
        :return:                True, if the rule learner is able to predict probability estimates, False otherwise
        """
        return self.get_rule_learner_ptr().canPredictProbabilities(
            dereference(feature_matrix.get_row_wise_feature_matrix_ptr()), num_labels)

    def create_probability_predictor(self, RowWiseFeatureMatrix feature_matrix not None, RuleModel rule_model not None,
                                     LabelSpaceInfo label_space_info not None,
                                     uint32 num_labels) -> ProbabilityPredictor:
        """
        Creates and returns a predictor that may be used to predict probability estimates for given query examples. If
        the prediction of probability estimates is not supported by the rule learner, a `RuntimeError` is thrown.

        :param feature_matrix:      A `RowWiseFeatureMatrix` that provides row-wise access to the feature values of the
                                    query examples
        :param rule_model:          The `RuleModel` that should be used to obtain predictions
        :param label_space_info:    The `LabelSpaceInfo` that provides information about the label space that may be
                                    used as a basis for obtaining predictions
        :param num_labels:          The number of labels to predict for
        :return:                    A `ProbabilityPredictor` that may be used to predict probability estimates for the
                                    given query examples
        """
        cdef unique_ptr[IProbabilityPredictor] predictor_ptr = move(self.get_rule_learner_ptr().createProbabilityPredictor(
            dereference(feature_matrix.get_row_wise_feature_matrix_ptr()),
            dereference(rule_model.get_rule_model_ptr()),
            dereference(label_space_info.get_label_space_info_ptr()),
            num_labels))
        cdef ProbabilityPredictor probability_predictor = ProbabilityPredictor.__new__(ProbabilityPredictor)
        probability_predictor.predictor_ptr = move(predictor_ptr)
        return probability_predictor
