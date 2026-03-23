"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""

import pytest

from ..cmd_runner import CmdRunner
from ..datasets import Dataset
from ..integration_tests import IntegrationTests

from mlrl.testbed.experiments.input.dataset.splitters.arguments import DatasetSplitterArguments
from mlrl.testbed.experiments.state import ExperimentMode

from mlrl.util.options import Options


class MlrlTestbedIntegrationTestsMixin(IntegrationTests):
    """
    A mixin for integration tests for the mlrl-testbed package.
    """

    def test_print_all(self):
        test_name = 'print-all'
        builder = self.create_cmd_builder().save_meta_data().print_all()
        CmdRunner(builder).run(test_name, wipe_after=False)
        builder = self.create_cmd_builder().set_mode(ExperimentMode.READ).print_all()
        CmdRunner(builder).run(test_name, wipe_before=False)

    def test_save_all(self):
        test_name = 'save-all'
        builder = self.create_cmd_builder().save_meta_data().save_all()
        CmdRunner(builder).run(test_name, wipe_after=False)
        builder = self.create_cmd_builder().set_mode(ExperimentMode.READ).save_all()
        CmdRunner(builder).run(test_name, wipe_before=False)

    def test_cancel_if_outputs_exist(self):
        test_name = 'cancel-if-outputs-exist'
        builder = self.create_cmd_builder()
        CmdRunner(builder).run(test_name, wipe_after=False, compare_output=False)
        CmdRunner(builder).run(test_name, wipe_before=False)

    def test_batch_mode(self):
        test_name = 'batch-mode'
        builder = self.create_cmd_builder().save_meta_data().set_mode(ExperimentMode.BATCH).save_all()
        CmdRunner(builder).run(test_name, wipe_after=False)
        builder = self.create_cmd_builder().set_mode(ExperimentMode.READ).save_all()
        CmdRunner(builder).run(test_name, wipe_before=False)

    def test_batch_mode_cancel_if_outputs_exist(self):
        test_name = 'batch-mode-cancel-if-outputs-exist'
        builder = self.create_cmd_builder().set_mode(ExperimentMode.BATCH)
        CmdRunner(builder).run(test_name, wipe_after=False, compare_output=False)
        CmdRunner(builder).run(test_name, wipe_before=False)

    def test_batch_mode_separate_folds(self):
        test_name = 'batch-mode-separate-folds'
        builder = (
            self.create_cmd_builder()
            .save_meta_data()
            .set_mode(ExperimentMode.BATCH)
            .save_all()
            .data_split(
                DatasetSplitterArguments.VALUE_CROSS_VALIDATION,
                options=Options({DatasetSplitterArguments.OPTION_NUM_FOLDS: 2}),
            )
        )
        CmdRunner(builder).run(test_name, wipe_after=False)
        builder = self.create_cmd_builder().set_mode(ExperimentMode.READ).save_all()
        CmdRunner(builder).run(test_name, wipe_before=False)

    def test_batch_mode_list(self):
        builder = self.create_cmd_builder().set_mode(ExperimentMode.BATCH, '--list')
        CmdRunner(builder).run('batch-mode-list')

    def test_batch_mode_slurm(self):
        builder = self.create_cmd_builder().save_meta_data().set_mode(ExperimentMode.BATCH).set_runner('slurm')
        CmdRunner(builder).run('batch-mode-slurm')

    def test_batch_mode_separate_folds_slurm(self):
        builder = (
            self.create_cmd_builder()
            .save_meta_data()
            .set_mode(ExperimentMode.BATCH)
            .set_runner('slurm')
            .data_split(
                DatasetSplitterArguments.VALUE_CROSS_VALIDATION,
                options=Options({DatasetSplitterArguments.OPTION_NUM_FOLDS: 2}),
            )
        )
        CmdRunner(builder).run('batch-mode-separate-folds-slurm')

    def test_run_mode(self):
        test_name = 'run-mode'
        builder = self.create_cmd_builder().save_meta_data()
        CmdRunner(builder).run(test_name, wipe_after=False)
        builder = self.create_cmd_builder().set_mode(ExperimentMode.RUN).save_all().print_all()
        CmdRunner(builder).run(test_name, wipe_before=False)

    def test_meka_format(self, dataset: Dataset):
        builder = self.create_cmd_builder(dataset=dataset.meka).print_evaluation(False)
        CmdRunner(builder).run('meka-format')

    def test_svm_format(self, dataset: Dataset):
        builder = self.create_cmd_builder(dataset=dataset.svm).print_evaluation(False)
        CmdRunner(builder).run('svm-format')

    def test_meta_data(self):
        test_name = 'meta_data'
        builder = (
            self.create_cmd_builder()
            .save_meta_data()
            .print_evaluation(False)
            .save_evaluation(False)
            .print_meta_data()
            .save_meta_data()
        )
        CmdRunner(builder).run(test_name, wipe_after=False)
        builder = (
            self.create_cmd_builder()
            .set_mode(ExperimentMode.READ)
            .print_evaluation(False)
            .save_evaluation(False)
            .print_meta_data()
            .save_meta_data()
        )
        CmdRunner(builder).run(test_name, wipe_before=False)

    @pytest.mark.parametrize('predefined', [False, True])
    def test_evaluation(self, predefined: bool, dataset: Dataset):
        test_name = 'evaluation' + ('-predefined' if predefined else '')
        builder = (
            self.create_cmd_builder(dataset=dataset.default + ('-predefined' if predefined else ''))
            .save_meta_data()
            .print_evaluation()
            .save_evaluation()
        )
        CmdRunner(builder).run(test_name, wipe_after=False)
        builder = self.create_cmd_builder().set_mode(ExperimentMode.READ).print_evaluation().save_evaluation()
        CmdRunner(builder).run(test_name, wipe_before=False)

    def test_evaluation_training_data(self, dataset: Dataset):
        test_name = 'evaluation_training-data'
        builder = (
            self.create_cmd_builder(dataset=dataset.default)
            .save_meta_data()
            .predict_for_training_data()
            .print_evaluation()
            .save_evaluation()
        )
        CmdRunner(builder).run(test_name, wipe_after=False)
        builder = self.create_cmd_builder().set_mode(ExperimentMode.READ).print_evaluation().save_evaluation()
        CmdRunner(builder).run(test_name, wipe_before=False)


def test_predictions(self, dataset: Dataset):
    test_name = 'predictions'
    builder = (
        self.create_cmd_builder(dataset=dataset.default)
        .save_meta_data()
        .print_evaluation(False)
        .save_evaluation(False)
        .print_predictions()
        .print_ground_truth()
        .save_predictions()
        .save_ground_truth()
    )
    CmdRunner(builder).run(test_name, wipe_after=False)
    builder = (
        self.create_cmd_builder()
        .set_mode(ExperimentMode.READ)
        .print_evaluation(False)
        .save_evaluation(False)
        .print_predictions()
        .save_predictions()
        .print_ground_truth()
        .save_ground_truth()
    )
    CmdRunner(builder).run(test_name, wipe_before=False)


def test_predictions_training_data(self, dataset: Dataset):
    test_name = 'predictions_training-data'
    builder = (
        self.create_cmd_builder(dataset=dataset.default)
        .save_meta_data()
        .predict_for_training_data()
        .print_evaluation(False)
        .save_evaluation(False)
        .print_predictions()
        .print_ground_truth()
        .save_predictions()
        .save_ground_truth()
    )
    CmdRunner(builder).run(test_name, wipe_after=False)
    builder = (
        self.create_cmd_builder()
        .set_mode(ExperimentMode.READ)
        .print_evaluation(False)
        .save_evaluation(False)
        .print_predictions()
        .save_predictions()
    )
    CmdRunner(builder).run(test_name, wipe_before=False)


def test_prediction_characteristics(self, dataset: Dataset):
    test_name = 'prediction-characteristics'
    builder = (
        self.create_cmd_builder(dataset=dataset.default)
        .save_meta_data()
        .print_evaluation(False)
        .save_evaluation(False)
        .print_prediction_characteristics()
        .save_prediction_characteristics()
    )
    CmdRunner(builder).run(test_name, wipe_after=False)
    builder = (
        self.create_cmd_builder()
        .set_mode(ExperimentMode.READ)
        .print_evaluation(False)
        .save_evaluation(False)
        .print_prediction_characteristics()
        .save_prediction_characteristics()
    )
    CmdRunner(builder).run(test_name, wipe_before=False)


def test_prediction_characteristics_training_data(self, dataset: Dataset):
    test_name = 'prediction-characteristics_training-data'
    builder = (
        self.create_cmd_builder(dataset=dataset.default)
        .save_meta_data()
        .predict_for_training_data()
        .print_evaluation(False)
        .save_evaluation(False)
        .print_prediction_characteristics()
        .save_prediction_characteristics()
    )
    CmdRunner(builder).run(test_name, wipe_after=False)
    builder = (
        self.create_cmd_builder()
        .set_mode(ExperimentMode.READ)
        .print_evaluation(False)
        .save_evaluation(False)
        .print_prediction_characteristics()
        .save_prediction_characteristics()
    )
    CmdRunner(builder).run(test_name, wipe_before=False)


def test_data_characteristics(self, dataset: Dataset):
    test_name = 'data-characteristics'
    builder = (
        self.create_cmd_builder(dataset=dataset.default)
        .save_meta_data()
        .print_evaluation(False)
        .save_evaluation(False)
        .print_data_characteristics()
        .save_data_characteristics()
    )
    CmdRunner(builder).run(test_name, wipe_after=False)
    builder = (
        self.create_cmd_builder()
        .set_mode(ExperimentMode.READ)
        .print_evaluation(False)
        .save_evaluation(False)
        .print_data_characteristics()
        .save_data_characteristics()
    )
    CmdRunner(builder).run(test_name, wipe_before=False)


def test_parameters(self, dataset: Dataset):
    test_name = 'parameters'
    builder = (
        self.create_cmd_builder(dataset=dataset.default)
        .save_meta_data()
        .print_evaluation(False)
        .save_evaluation(False)
        .print_parameters()
        .save_parameters()
        .load_parameters()
    )
    CmdRunner(builder).run(test_name, wipe_after=False)
    builder = (
        self.create_cmd_builder()
        .set_mode(ExperimentMode.READ)
        .print_evaluation(False)
        .save_evaluation(False)
        .print_parameters()
        .save_parameters()
    )
    CmdRunner(builder).run(test_name, wipe_before=False)
