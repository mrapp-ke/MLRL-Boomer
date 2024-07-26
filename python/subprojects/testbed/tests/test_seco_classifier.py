"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from os import path
from typing import Any

from test_common import DATASET_EMOTIONS, DIR_OUT, RULE_PRUNING_IREP, CmdBuilder
from test_common_classification import ClassificationCmdBuilder, ClassificationIntegrationTests

HEURISTIC_ACCURACY = 'accuracy'

HEURISTIC_PRECISION = 'precision'

HEURISTIC_LAPLACE = 'laplace'

HEURISTIC_RECALL = 'recall'

HEURISTIC_WRA = 'weighted-relative-accuracy'

HEURISTIC_F_MEASURE = 'f-measure'

HEURISTIC_M_ESTIMATE = 'm-estimate'

HEAD_TYPE_SINGLE = 'single'

HEAD_TYPE_PARTIAL = 'partial'

LIFT_FUNCTION_NO = 'none'

LIFT_FUNCTION_PEAK = 'peak'

LIFT_FUNCTION_KLN = 'kln'


class SeCoClassifierCmdBuilder(ClassificationCmdBuilder):
    """
    A builder that allows to configure a command for running the separate-and-conquer (SeCo) algorithm.
    """

    def __init__(self, callback: CmdBuilder.AssertionCallback, dataset: str = DATASET_EMOTIONS):
        super().__init__(callback,
                         expected_output_dir=path.join(DIR_OUT, 'seco-classifier'),
                         model_file_name='seco',
                         runnable_module_name='mlrl.seco',
                         dataset=dataset)

    def heuristic(self, heuristic: str = HEURISTIC_F_MEASURE):
        """
        Configures the algorithm to use a specific heuristic for learning rules.

        :param heuristic:   The name of the heuristic that should be used for learning rules
        :return:            The builder itself
        """
        self.args.append('--heuristic')
        self.args.append(heuristic)
        return self

    def pruning_heuristic(self, heuristic: str = HEURISTIC_ACCURACY):
        """
        Configures the algorithm to use a specific heuristic for pruning rules.

        :param heuristic:   The name of the heuristic that should be used for pruning rules
        :return:            The builder itself
        """
        self.args.append('--pruning-heuristic')
        self.args.append(heuristic)
        return self

    def head_type(self, head_type: str = HEAD_TYPE_SINGLE):
        """
        Configures the algorithm to use a specific type of rule heads.

        :param head_type:   The type of rule heads to be used
        :return:            The builder itself
        """
        self.args.append('--head-type')
        self.args.append(head_type)
        return self

    def lift_function(self, lift_function: str = LIFT_FUNCTION_PEAK):
        """
        Configures the algorithm to use a specific lift function for the induction of rules with partial heads.

        :param lift_function:   The name of the lift function that should be used for the induction of rules with
                                partial heads
        :return:                The builder itself
        """
        self.args.append('--lift-function')
        self.args.append(lift_function)
        return self


class SeCoClassifierIntegrationTests(ClassificationIntegrationTests):
    """
    Defines a series of integration tests for the separate-and-conquer (SeCo) algorithm for classification problems.
    """

    def __init__(self, methodName='runTest'):
        super().__init__(methodName=methodName)

    def _create_cmd_builder(self, dataset: str = DATASET_EMOTIONS) -> Any:
        return SeCoClassifierCmdBuilder(self, dataset=dataset)

    def test_heuristic_accuracy(self):
        """
        Tests the SeCo algorithm when using the accuracy heuristic for learning rules.
        """
        builder = self._create_cmd_builder() \
            .heuristic(HEURISTIC_ACCURACY)
        builder.run_cmd('heuristic_accuracy')

    def test_heuristic_precision(self):
        """
        Tests the SeCo algorithm when using the precision heuristic for learning rules.
        """
        builder = self._create_cmd_builder() \
            .heuristic(HEURISTIC_PRECISION)
        builder.run_cmd('heuristic_precision')

    def test_heuristic_recall(self):
        """
        Tests the SeCo algorithm when using the recall heuristic for learning rules.
        """
        builder = self._create_cmd_builder() \
            .heuristic(HEURISTIC_RECALL)
        builder.run_cmd('heuristic_recall')

    def test_heuristic_laplace(self):
        """
        Tests the SeCo algorithm when using the Laplace heuristic for learning rules.
        """
        builder = self._create_cmd_builder() \
            .heuristic(HEURISTIC_LAPLACE)
        builder.run_cmd('heuristic_laplace')

    def test_heuristic_wra(self):
        """
        Tests the SeCo algorithm when using the WRA heuristic for learning rules.
        """
        builder = self._create_cmd_builder() \
            .heuristic(HEURISTIC_WRA)
        builder.run_cmd('heuristic_wra')

    def test_heuristic_f_measure(self):
        """
        Tests the SeCo algorithm when using the F-measure heuristic for learning rules.
        """
        builder = self._create_cmd_builder() \
            .heuristic(HEURISTIC_F_MEASURE)
        builder.run_cmd('heuristic_f-measure')

    def test_heuristic_m_estimate(self):
        """
        Tests the SeCo algorithm when using the m-estimate heuristic for learning rules.
        """
        builder = self._create_cmd_builder() \
            .heuristic(HEURISTIC_M_ESTIMATE)
        builder.run_cmd('heuristic_m-estimate')

    def test_pruning_heuristic_accuracy(self):
        """
        Tests the SeCo algorithm when using the accuracy heuristic for pruning rules.
        """
        builder = self._create_cmd_builder() \
            .rule_pruning(RULE_PRUNING_IREP) \
            .pruning_heuristic(HEURISTIC_ACCURACY)
        builder.run_cmd('pruning-heuristic_accuracy')

    def test_pruning_heuristic_precision(self):
        """
        Tests the SeCo algorithm when using the precision heuristic for pruning rules.
        """
        builder = self._create_cmd_builder() \
            .rule_pruning(RULE_PRUNING_IREP) \
            .pruning_heuristic(HEURISTIC_PRECISION)
        builder.run_cmd('pruning-heuristic_precision')

    def test_pruning_heuristic_recall(self):
        """
        Tests the SeCo algorithm when using the recall heuristic for pruning rules.
        """
        builder = self._create_cmd_builder() \
            .rule_pruning(RULE_PRUNING_IREP) \
            .pruning_heuristic(HEURISTIC_RECALL)
        builder.run_cmd('pruning-heuristic_recall')

    def test_pruning_heuristic_laplace(self):
        """
        Tests the SeCo algorithm when using the Laplace heuristic for pruning rules.
        """
        builder = self._create_cmd_builder() \
            .rule_pruning(RULE_PRUNING_IREP) \
            .pruning_heuristic(HEURISTIC_LAPLACE)
        builder.run_cmd('pruning-heuristic_laplace')

    def test_pruning_heuristic_wra(self):
        """
        Tests the SeCo algorithm when using the WRA heuristic for pruning rules.
        """
        builder = self._create_cmd_builder() \
            .rule_pruning(RULE_PRUNING_IREP) \
            .pruning_heuristic(HEURISTIC_WRA)
        builder.run_cmd('pruning-heuristic_wra')

    def test_pruning_heuristic_f_measure(self):
        """
        Tests the SeCo algorithm when using the F-measure heuristic for pruning rules.
        """
        builder = self._create_cmd_builder() \
            .rule_pruning(RULE_PRUNING_IREP) \
            .pruning_heuristic(HEURISTIC_F_MEASURE)
        builder.run_cmd('pruning-heuristic_f-measure')

    def test_pruning_heuristic_m_estimate(self):
        """
        Tests the SeCo algorithm when using the m-estimate heuristic for pruning rules.
        """
        builder = self._create_cmd_builder() \
            .rule_pruning(RULE_PRUNING_IREP) \
            .pruning_heuristic(HEURISTIC_M_ESTIMATE)
        builder.run_cmd('pruning-heuristic_m-estimate')

    def test_single_output_heads(self):
        """
        Tests the SeCo algorithm when inducing rules with single-output heads.
        """
        builder = self._create_cmd_builder() \
            .head_type(HEAD_TYPE_SINGLE) \
            .print_model_characteristics()
        builder.run_cmd('single-output-heads')

    def test_partial_heads_no_lift_function(self):
        """
        Tests the SeCo algorithm when inducing partial rules using no lift function.
        """
        builder = self._create_cmd_builder() \
            .head_type(HEAD_TYPE_PARTIAL) \
            .lift_function(LIFT_FUNCTION_NO) \
            .print_model_characteristics()
        builder.run_cmd('partial-heads_no-lift-function')

    def test_partial_heads_peak_lift_function(self):
        """
        Tests the SeCo algorithm when inducing partial rules using the peak lift function.
        """
        builder = self._create_cmd_builder() \
            .head_type(HEAD_TYPE_PARTIAL) \
            .lift_function(LIFT_FUNCTION_PEAK) \
            .print_model_characteristics()
        builder.run_cmd('partial-heads_peak-lift-function')

    def test_partial_heads_kln_lift_function(self):
        """
        Tests the SeCo algorithm when inducing partial rules using the KLN lift function.
        """
        builder = self._create_cmd_builder() \
            .head_type(HEAD_TYPE_PARTIAL) \
            .lift_function(LIFT_FUNCTION_KLN) \
            .print_model_characteristics()
        builder.run_cmd('partial-heads_kln-lift-function')
