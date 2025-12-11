"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
# pylint: disable=missing-function-docstring
from typing import Any, Optional, override

import pytest

from sklearn.utils.estimator_checks import check_estimator

from ..common.cmd_runner import CmdRunner
from ..common.datasets import Dataset
from ..common.integration_tests_classification import ClassificationIntegrationTests
from ..common.integration_tests_rule_learners import ClassificationRuleLearnerIntegrationTestsMixin
from .cmd_builder_classification import BoomerClassifierCmdBuilder
from .integration_tests import BoomerIntegrationTestsMixin

from mlrl.common.config.parameters import BINNING_EQUAL_WIDTH, SAMPLING_STRATIFIED_EXAMPLE_WISE, \
    SAMPLING_STRATIFIED_OUTPUT_WISE, GlobalPruningParameter
from mlrl.common.learners import SparsePolicy

from mlrl.boosting.config.parameters import OPTION_BASED_ON_PROBABILITIES, PROBABILITY_CALIBRATION_ISOTONIC, \
    BinaryPredictorParameter, ClassificationLossParameter, HeadTypeParameter, ProbabilityPredictorParameter, \
    StatisticTypeParameter
from mlrl.boosting.learners import BoomerClassifier

from mlrl.testbed.experiments.prediction_type import PredictionType
from mlrl.testbed.experiments.state import ExperimentMode

from mlrl.util.cli import NONE
from mlrl.util.options import BooleanOption, Options


@pytest.mark.boosting
@pytest.mark.classification
class TestBoomerClassifier(ClassificationIntegrationTests, ClassificationRuleLearnerIntegrationTestsMixin,
                           BoomerIntegrationTestsMixin):
    """
    Defines a series of integration tests for the BOOMER algorithm for classification problems.
    """

    @override
    def _create_cmd_builder(self, dataset: str = Dataset.EMOTIONS) -> Any:
        return BoomerClassifierCmdBuilder(dataset=dataset)

    def test_scikit_learn_compatibility(self):
        check_estimator(BoomerClassifier(),
                        expected_failed_checks={
                            'check_classifiers_train': 'Fails because model is too large to pickle',
                            'check_estimators_pickle': 'Fails because model is too large to pickle',
                            'check_readonly_memmap_input': 'Fails because model is too large to pickle',
                        })

    @pytest.mark.parametrize('prediction_type', [
        PredictionType.SCORES,
        PredictionType.PROBABILITIES,
    ])
    def test_single_label(self, prediction_type: PredictionType, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.single_output) \
            .prediction_type(prediction_type) \
            .print_evaluation()
        CmdRunner(builder).run(f'single-label-{prediction_type}')

    @pytest.mark.parametrize('loss', [
        ClassificationLossParameter.LOSS_LOGISTIC_DECOMPOSABLE,
        pytest.param(ClassificationLossParameter.LOSS_LOGISTIC_NON_DECOMPOSABLE, marks=pytest.mark.flaky(reruns=5)),
        ClassificationLossParameter.LOSS_SQUARED_HINGE_DECOMPOSABLE,
        pytest.param(ClassificationLossParameter.LOSS_SQUARED_HINGE_NON_DECOMPOSABLE,
                     marks=pytest.mark.flaky(reruns=5)),
    ])
    @pytest.mark.parametrize('statistic_type', [
        StatisticTypeParameter.STATISTIC_TYPE_FLOAT32,
        StatisticTypeParameter.STATISTIC_TYPE_FLOAT64,
    ])
    def test_loss(self, loss: str, statistic_type: str):
        builder = self._create_cmd_builder() \
            .loss(loss) \
            .statistic_type(statistic_type)
        CmdRunner(builder).run(f'loss-{loss}_{statistic_type}-statistics')

    @pytest.mark.parametrize(
        'binary_predictor, binary_predictor_options, marginal_probability_calibration, joint_probability_calibration, '
        + 'label_vectors, prediction_format', [
            (BinaryPredictorParameter.BINARY_PREDICTOR_OUTPUT_WISE, Options(), None, None, None, None),
            (BinaryPredictorParameter.BINARY_PREDICTOR_OUTPUT_WISE,
             Options({
                 OPTION_BASED_ON_PROBABILITIES: BooleanOption.TRUE,
             }), PROBABILITY_CALIBRATION_ISOTONIC, None, None, None),
            (BinaryPredictorParameter.BINARY_PREDICTOR_OUTPUT_WISE, Options(), None, None, None,
             SparsePolicy.FORCE_SPARSE),
            (BinaryPredictorParameter.BINARY_PREDICTOR_EXAMPLE_WISE, Options(), None, None, True, None),
            (BinaryPredictorParameter.BINARY_PREDICTOR_EXAMPLE_WISE,
             Options({
                 OPTION_BASED_ON_PROBABILITIES: BooleanOption.TRUE,
             }), PROBABILITY_CALIBRATION_ISOTONIC, PROBABILITY_CALIBRATION_ISOTONIC, True, None),
            (BinaryPredictorParameter.BINARY_PREDICTOR_EXAMPLE_WISE, Options(), None, None, True,
             SparsePolicy.FORCE_SPARSE),
            (BinaryPredictorParameter.BINARY_PREDICTOR_GFM, Options(), PROBABILITY_CALIBRATION_ISOTONIC,
             PROBABILITY_CALIBRATION_ISOTONIC, True, None),
            (BinaryPredictorParameter.BINARY_PREDICTOR_GFM, Options(), PROBABILITY_CALIBRATION_ISOTONIC,
             PROBABILITY_CALIBRATION_ISOTONIC, True, SparsePolicy.FORCE_SPARSE),
        ])
    def test_predictor_binary(self, binary_predictor: str, binary_predictor_options: Options,
                              marginal_probability_calibration: Optional[str],
                              joint_probability_calibration: Optional[str], label_vectors: Optional[bool],
                              prediction_format: Optional[str]):
        test_name = f'predictor-binary-{binary_predictor}' + (f'_{prediction_format}' if prediction_format else '') + (
            f'_{binary_predictor_options}' if binary_predictor_options else '')
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration(marginal_probability_calibration) \
            .print_marginal_probability_calibration_model(True if marginal_probability_calibration else None) \
            .save_marginal_probability_calibration_model(True if marginal_probability_calibration else None) \
            .joint_probability_calibration(joint_probability_calibration) \
            .print_joint_probability_calibration_model(True if joint_probability_calibration else None) \
            .save_joint_probability_calibration_model(True if joint_probability_calibration else None) \
            .binary_predictor(binary_predictor, options=binary_predictor_options) \
            .print_predictions() \
            .print_ground_truth() \
            .print_label_vectors(label_vectors) \
            .prediction_format(prediction_format)

        if marginal_probability_calibration or joint_probability_calibration:
            builder.save_models()
            builder.load_models()

        CmdRunner(builder).run(test_name,
                               wipe_after=not marginal_probability_calibration and not joint_probability_calibration)

        if marginal_probability_calibration or joint_probability_calibration:
            builder = self._create_cmd_builder() \
                .set_mode(ExperimentMode.READ) \
                .print_evaluation(False) \
                .save_evaluation(False) \
                .print_marginal_probability_calibration_model(True if marginal_probability_calibration else None) \
                .save_marginal_probability_calibration_model(True if marginal_probability_calibration else None) \
                .print_joint_probability_calibration_model(True if joint_probability_calibration else None) \
                .save_joint_probability_calibration_model(True if joint_probability_calibration else None)
            CmdRunner(builder).run(test_name, wipe_before=False)

    @pytest.mark.parametrize(
        'binary_predictor, binary_predictor_options, marginal_probability_calibration, joint_probability_calibration, '
        + 'prediction_format', [
            (BinaryPredictorParameter.BINARY_PREDICTOR_OUTPUT_WISE, Options(), None, None, None),
            (BinaryPredictorParameter.BINARY_PREDICTOR_OUTPUT_WISE,
             Options({
                 OPTION_BASED_ON_PROBABILITIES: BooleanOption.TRUE,
             }), PROBABILITY_CALIBRATION_ISOTONIC, None, None),
            (BinaryPredictorParameter.BINARY_PREDICTOR_OUTPUT_WISE, Options(), None, None, SparsePolicy.FORCE_SPARSE),
            (BinaryPredictorParameter.BINARY_PREDICTOR_EXAMPLE_WISE, Options(), None, None, None),
            (BinaryPredictorParameter.BINARY_PREDICTOR_EXAMPLE_WISE,
             Options({
                 OPTION_BASED_ON_PROBABILITIES: BooleanOption.TRUE,
             }), PROBABILITY_CALIBRATION_ISOTONIC, PROBABILITY_CALIBRATION_ISOTONIC, None),
            (BinaryPredictorParameter.BINARY_PREDICTOR_EXAMPLE_WISE, Options(), None, None, SparsePolicy.FORCE_SPARSE),
            (BinaryPredictorParameter.BINARY_PREDICTOR_GFM, Options(), PROBABILITY_CALIBRATION_ISOTONIC,
             PROBABILITY_CALIBRATION_ISOTONIC, None),
            (BinaryPredictorParameter.BINARY_PREDICTOR_GFM, Options(), PROBABILITY_CALIBRATION_ISOTONIC,
             PROBABILITY_CALIBRATION_ISOTONIC, SparsePolicy.FORCE_SPARSE),
        ])
    def test_predictor_binary_incremental(self, binary_predictor: str, binary_predictor_options: Options,
                                          marginal_probability_calibration: Optional[str],
                                          joint_probability_calibration: Optional[str],
                                          prediction_format: Optional[str]):
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration(marginal_probability_calibration) \
            .print_marginal_probability_calibration_model(True if marginal_probability_calibration else None) \
            .save_marginal_probability_calibration_model(True if marginal_probability_calibration else None) \
            .joint_probability_calibration(joint_probability_calibration) \
            .print_joint_probability_calibration_model(True if joint_probability_calibration else None) \
            .save_joint_probability_calibration_model(True if joint_probability_calibration else None) \
            .binary_predictor(binary_predictor, options=binary_predictor_options) \
            .incremental_evaluation() \
            .print_evaluation() \
            .save_evaluation() \
            .prediction_format(prediction_format)

        if marginal_probability_calibration or joint_probability_calibration:
            builder.save_models()
            builder.load_models()

        CmdRunner(builder).run(f'predictor-binary-{binary_predictor}'
                               + (f'_{prediction_format}' if prediction_format else '')
                               + (f'_{binary_predictor_options}' if binary_predictor_options else '') + '_incremental')

    def test_predictor_score_output_wise(self):
        builder = self._create_cmd_builder() \
            .prediction_type(PredictionType.SCORES) \
            .print_predictions() \
            .print_ground_truth()
        CmdRunner(builder).run('predictor-score-output-wise')

    def test_predictor_score_output_wise_incremental(self):
        builder = self._create_cmd_builder() \
            .prediction_type(PredictionType.SCORES) \
            .incremental_evaluation() \
            .print_evaluation() \
            .save_evaluation()
        CmdRunner(builder).run('predictor-score-output-wise_incremental')

    @pytest.mark.parametrize(
        'probability_predictor, marginal_probability_calibration, joint_probability_calibration, print_label_vectors', [
            (ProbabilityPredictorParameter.PROBABILITY_PREDICTOR_OUTPUT_WISE, PROBABILITY_CALIBRATION_ISOTONIC, None,
             None),
            (ProbabilityPredictorParameter.PROBABILITY_PREDICTOR_MARGINALIZED, PROBABILITY_CALIBRATION_ISOTONIC,
             PROBABILITY_CALIBRATION_ISOTONIC, True),
        ])
    def test_predictor_probability(self, probability_predictor: str, marginal_probability_calibration: Optional[str],
                                   joint_probability_calibration: Optional[str], print_label_vectors: Optional[bool]):
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration(marginal_probability_calibration) \
            .print_marginal_probability_calibration_model(True if marginal_probability_calibration else None) \
            .save_marginal_probability_calibration_model(True if marginal_probability_calibration else None) \
            .joint_probability_calibration(joint_probability_calibration) \
            .print_joint_probability_calibration_model(True if joint_probability_calibration else None) \
            .save_joint_probability_calibration_model(True if joint_probability_calibration else None) \
            .save_evaluation(False) \
            .prediction_type(PredictionType.PROBABILITIES) \
            .probability_predictor(probability_predictor) \
            .print_predictions() \
            .print_ground_truth() \
            .print_label_vectors(print_label_vectors) \
            .save_models() \
            .load_models()
        CmdRunner(builder).run(f'predictor-probability-{probability_predictor}')

    @pytest.mark.parametrize('probability_predictor, marginal_probability_calibration, joint_probability_calibration', [
        (ProbabilityPredictorParameter.PROBABILITY_PREDICTOR_OUTPUT_WISE, PROBABILITY_CALIBRATION_ISOTONIC, None),
        (ProbabilityPredictorParameter.PROBABILITY_PREDICTOR_MARGINALIZED, PROBABILITY_CALIBRATION_ISOTONIC,
         PROBABILITY_CALIBRATION_ISOTONIC),
    ])
    def test_predictor_probability_incremental(self, probability_predictor: str,
                                               marginal_probability_calibration: Optional[str],
                                               joint_probability_calibration: Optional[str]):
        builder = self._create_cmd_builder() \
            .marginal_probability_calibration(marginal_probability_calibration) \
            .print_marginal_probability_calibration_model(True if marginal_probability_calibration else None) \
            .save_marginal_probability_calibration_model(True if marginal_probability_calibration else None) \
            .joint_probability_calibration(joint_probability_calibration) \
            .print_joint_probability_calibration_model(True if joint_probability_calibration else None) \
            .save_joint_probability_calibration_model(True if joint_probability_calibration else None) \
            .prediction_type(PredictionType.PROBABILITIES) \
            .probability_predictor(probability_predictor) \
            .incremental_evaluation() \
            .print_evaluation() \
            .save_evaluation() \
            .save_models() \
            .load_models()
        CmdRunner(builder).run(f'predictor-probability-{probability_predictor}_incremental')

    @pytest.mark.parametrize('output_format', [
        SparsePolicy.FORCE_DENSE,
        SparsePolicy.FORCE_SPARSE,
    ])
    def test_statistics_sparse(self, output_format: str, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.numerical_sparse) \
            .sparse_statistic_format() \
            .output_format(output_format) \
            .default_rule(False) \
            .loss(ClassificationLossParameter.LOSS_SQUARED_HINGE_DECOMPOSABLE) \
            .head_type(HeadTypeParameter.HEAD_TYPE_SINGLE)
        CmdRunner(builder).run(f'statistics-sparse_output-format-{output_format}')

    @pytest.mark.parametrize('head_type, label_binning', [
        (HeadTypeParameter.HEAD_TYPE_SINGLE, None),
        (HeadTypeParameter.HEAD_TYPE_COMPLETE, None),
        (HeadTypeParameter.HEAD_TYPE_COMPLETE, BINNING_EQUAL_WIDTH),
        (HeadTypeParameter.HEAD_TYPE_PARTIAL_FIXED, None),
        (HeadTypeParameter.HEAD_TYPE_PARTIAL_FIXED, BINNING_EQUAL_WIDTH),
        (HeadTypeParameter.HEAD_TYPE_PARTIAL_DYNAMIC, None),
        (HeadTypeParameter.HEAD_TYPE_PARTIAL_DYNAMIC, BINNING_EQUAL_WIDTH),
    ])
    @pytest.mark.parametrize('statistic_type', [
        StatisticTypeParameter.STATISTIC_TYPE_FLOAT32,
        StatisticTypeParameter.STATISTIC_TYPE_FLOAT64,
    ])
    def test_decomposable_head_type(self, head_type: str, label_binning: Optional[str], statistic_type: str):
        builder = self._create_cmd_builder() \
            .loss(ClassificationLossParameter.LOSS_LOGISTIC_DECOMPOSABLE) \
            .statistic_type(statistic_type) \
            .head_type(head_type) \
            .label_binning(label_binning) \
            .print_model_characteristics()
        CmdRunner(builder).run(f'decomposable-{head_type}-heads'
                               + (f'_{label_binning}-label-binning' if label_binning else '')
                               + f'_{statistic_type}-statistics')

    @pytest.mark.parametrize('head_type, label_binning', [
        (HeadTypeParameter.HEAD_TYPE_SINGLE, None),
        (HeadTypeParameter.HEAD_TYPE_COMPLETE, NONE),
        pytest.param(HeadTypeParameter.HEAD_TYPE_COMPLETE, BINNING_EQUAL_WIDTH, marks=pytest.mark.flaky(reruns=5)),
        (HeadTypeParameter.HEAD_TYPE_PARTIAL_FIXED, NONE),
        (HeadTypeParameter.HEAD_TYPE_PARTIAL_FIXED, BINNING_EQUAL_WIDTH),
        (HeadTypeParameter.HEAD_TYPE_PARTIAL_DYNAMIC, NONE),
        (HeadTypeParameter.HEAD_TYPE_PARTIAL_DYNAMIC, BINNING_EQUAL_WIDTH),
    ])
    @pytest.mark.parametrize('statistic_type', [
        StatisticTypeParameter.STATISTIC_TYPE_FLOAT32,
        StatisticTypeParameter.STATISTIC_TYPE_FLOAT64,
    ])
    def test_non_decomposable_head_type(self, head_type: str, label_binning: Optional[str], statistic_type: str):
        builder = self._create_cmd_builder() \
            .loss(ClassificationLossParameter.LOSS_LOGISTIC_NON_DECOMPOSABLE) \
            .statistic_type(statistic_type) \
            .head_type(head_type) \
            .label_binning(label_binning) \
            .print_model_characteristics()
        CmdRunner(builder).run(f'non-decomposable-{head_type}-heads'
                               + (f'_{label_binning}-label-binning' if label_binning else '')
                               + f'_{statistic_type}-statistics')

    @pytest.mark.parametrize('global_pruning', [
        GlobalPruningParameter.GLOBAL_PRUNING_POST,
        GlobalPruningParameter.GLOBAL_PRUNING_PRE,
    ])
    @pytest.mark.parametrize('holdout', [
        SAMPLING_STRATIFIED_OUTPUT_WISE,
        SAMPLING_STRATIFIED_EXAMPLE_WISE,
    ])
    def test_global_pruning_stratified_holdout(self, global_pruning: str, holdout: str):
        builder = self._create_cmd_builder() \
            .global_pruning(global_pruning) \
            .holdout(holdout) \
            .print_model_characteristics()
        CmdRunner(builder).run(f'{global_pruning}_{holdout}-holdout')
