"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from typing import Optional

from .cmd_builder import CmdBuilder

from mlrl.common.config.parameters import BINNING_EQUAL_WIDTH, SAMPLING_WITHOUT_REPLACEMENT, \
    PartitionSamplingParameter, PostOptimizationParameter, RuleInductionParameter, RulePruningParameter
from mlrl.common.learners import SparsePolicy


class RuleLearnerCmdBuilderMixin(CmdBuilder):
    """
    A mixin for all builders that allow to configure a command for running a rule learner.
    """

    def sparse_feature_value(self, sparse_feature_value: float = 0.0):
        """
        Configures the value that should be used for sparse elements in the feature matrix.

        :param sparse_feature_value:    The value that should be used for sparse elements in the feature matrix
        :return:                        The builder itself
        """
        self.args.append('--sparse-feature-value')
        self.args.append(str(sparse_feature_value))
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

    def print_model_characteristics(self, print_model_characteristics: bool = True):
        """
        Configures whether the characteristics of models should be printed on the console or not.

        :param print_model_characteristics: True, if the characteristics of models should be printed, False otherwise
        :return:                            The builder itself
        """
        self.args.append('--print-model-characteristics')
        self.args.append(str(print_model_characteristics).lower())
        return self

    def save_model_characteristics(self, save_model_characteristics: bool = True):
        """
        Configures whether the characteristics of models should be written to output files or not.

        :param save_model_characteristics:  True, if the characteristics of models should be written to output files,
                                            False otherwise
        :return:                            The builder itself
        """
        self.args.append('--save-model-characteristics')
        self.args.append(str(save_model_characteristics).lower())
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

    def save_rules(self, save_rules: bool = True):
        """
        Configures whether textual representations of the rules in a model should be written to output files or not.

        :param save_rules:  True, if textual representations of rules should be written to output files, False
                            otherwise
        :return:            The builder itself
        """
        self.args.append('--save-rules')
        self.args.append(str(save_rules).lower())
        return self

    def feature_format(self, feature_format: Optional[str] = SparsePolicy.FORCE_SPARSE):
        """
        Configures the format to be used for the feature values of training examples.

        :param feature_format:  The format to be used
        :return:                The builder itself
        """
        if feature_format:
            self.args.append('--feature-format')
            self.args.append(feature_format)

        return self

    def output_format(self, output_format: Optional[str] = SparsePolicy.FORCE_SPARSE):
        """
        Configures the format to be used for the ground truth of training examples.

        :param output_format:   The format to be used
        :return:                The builder itself
        """
        if output_format:
            self.args.append('--output-format')
            self.args.append(output_format)

        return self

    def prediction_format(self, prediction_format: Optional[str] = SparsePolicy.FORCE_SPARSE):
        """
        Configures the format to be used for predictions.

        :param prediction_format:   The format to be used
        :return:                    The builder itself
        """
        if prediction_format:
            self.args.append('--prediction-format')
            self.args.append(prediction_format)

        return self

    def instance_sampling(self, instance_sampling: Optional[str]):
        """
        Configures the rule learner to sample from the available training examples.

        :param instance_sampling:   The name of the sampling method that should be used
        :return:                    The builder itself
        """
        if instance_sampling:
            self.args.append('--instance-sampling')
            self.args.append(instance_sampling)

        return self

    def feature_sampling(self, feature_sampling: Optional[str] = SAMPLING_WITHOUT_REPLACEMENT):
        """
        Configures the rule learner to sample from the available features.

        :param feature_sampling:    The name of the sampling method that should be used
        :return:                    The builder itself
        """
        if feature_sampling:
            self.args.append('--feature-sampling')
            self.args.append(feature_sampling)

        return self

    def output_sampling(self, output_sampling: Optional[str] = SAMPLING_WITHOUT_REPLACEMENT):
        """
        Configures the rule learner to sample from the available outputs.

        :param output_sampling: The name of the sampling method that should be used
        :return:                The builder itself
        """
        if output_sampling:
            self.args.append('--output-sampling')
            self.args.append(output_sampling)

        return self

    def rule_pruning(self, rule_pruning: Optional[str] = RulePruningParameter.RULE_PRUNING_IREP):
        """
        Configures the rule learner to use a specific method for pruning individual rules.

        :param rule_pruning:    The name of the pruning method that should be used
        :return:                The builder itself
        """
        if rule_pruning:
            self.args.append('--rule-pruning')
            self.args.append(rule_pruning)

        return self

    def rule_induction(self, rule_induction: Optional[str] = RuleInductionParameter.RULE_INDUCTION_TOP_DOWN_GREEDY):
        """
        Configures the rule learner to use a specific algorithm for the induction of individual rules.

        :param rule_induction:  The name of the algorithm that should be used
        :return:                The builder itself
        """
        if rule_induction:
            self.args.append('--rule-induction')
            self.args.append(rule_induction)

        return self

    def post_optimization(self,
                          post_optimization: Optional[str] = PostOptimizationParameter.POST_OPTIMIZATION_SEQUENTIAL):
        """
        Configures the post-optimization method to be used by the algorithm.

        :param post_optimization:   The name of the method that should be used for post-optimization
        :return:                    The builder itself
        """
        if post_optimization:
            self.args.append('--post-optimization')
            self.args.append(post_optimization)

        return self

    def holdout(self, holdout: Optional[str] = PartitionSamplingParameter.PARTITION_SAMPLING_RANDOM):
        """
        Configures the algorithm to use a holdout set.

        :param holdout: The name of the sampling method that should be used to create the holdout set
        :return:        The builder itself
        """
        if holdout:
            self.args.append('--holdout')
            self.args.append(holdout)

        return self

    def feature_binning(self, feature_binning: Optional[str] = BINNING_EQUAL_WIDTH):
        """
        Configures the algorithm to use a specific method for feature binning.

        :param feature_binning: The name of the method that should be used for feature binning
        :return:                The builder itself
        """
        if feature_binning:
            self.args.append('--feature-binning')
            self.args.append(feature_binning)

        return self
