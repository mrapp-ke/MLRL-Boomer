"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing paths to files.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.state import ExperimentState


@dataclass
class FilePath:
    """
    The path to a file, data can be written to or read from.

    Attributes:
        directory:  The path to the directory, where the file is located
        file_name:  The name of the file
        suffix:     The suffix of the file (without leading dot) or None, if the suffix is unspecified
        context:    A `Context` to be used to determine the path
    """
    directory: Path
    file_name: str
    suffix: Optional[str]
    context: Context

    def resolve(self, state: ExperimentState) -> Path:
        """
        Determines and returns the path to the file to which output data should be written.

        :param state:   The state from which the output data has been generated
        :return:        The path to the file to which output data should be written
        """
        file_name = self.file_name

        if self.context.include_dataset_type:
            dataset_type = state.dataset_type

            if dataset_type:
                file_name += '_' + dataset_type

        if self.context.include_prediction_scope:
            prediction_result = state.prediction_result

            if prediction_result:
                prediction_scope = prediction_result.prediction_scope

                if not prediction_scope.is_global:
                    file_name += '_model-size-' + str(prediction_scope.model_size)

        if self.context.include_fold:
            folding_strategy = state.folding_strategy

            if folding_strategy and folding_strategy.is_cross_validation_used:
                fold = state.fold

                if fold:
                    file_name += '_fold-' + str(fold.index + 1)

        path = self.directory / Path(file_name)
        return path.with_suffix('.' + self.suffix) if self.suffix else path
