"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing paths to files.
"""
from os import path

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.state import ExperimentState


class FilePath:
    """
    The path to a file, data can be written to or read from.
    """

    def __init__(self, directory: str, file_name: str, suffix: str, context: Context):
        """
        :param directory:   The path to the directory, where the file is located
        :param file_name:   The name of the file
        :param suffix:      The suffix of the file (with leading dot)
        :param context:     A `Context` to be used to determine the path
        """
        self.directory = directory
        self.file_name = file_name
        self.suffix = suffix
        self.context = context

    def resolve(self, state: ExperimentState) -> str:
        """
        Determines and returns the path to the file to which output data should be written.

        :param state: The state from which the output data has been generated
        """
        file_name = self.file_name

        if self.context.include_dataset_type:
            dataset_type = state.dataset_type

            if dataset_type:
                file_name += '_' + dataset_type.value

        if self.context.include_prediction_scope:
            prediction_result = state.prediction_result

            if prediction_result:
                prediction_scope = prediction_result.prediction_scope

                if prediction_scope.is_global:
                    file_name += '_model-size-' + str(prediction_scope.model_size)

        if self.context.include_fold and state.folding_strategy.is_cross_validation_used:
            fold = state.fold

            if fold:
                file_name += '_fold-' + str(fold.index + 1)

        return path.join(self.directory, file_name + '.' + self.suffix)
