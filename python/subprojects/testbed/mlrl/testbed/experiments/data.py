"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing input or output data.
"""
from abc import ABC
from dataclasses import dataclass
from os import path

from mlrl.testbed.experiments.state import ExperimentState


class Data(ABC):
    """
    An abstract class for all classes that represent data that can be read from sources or written to sinks.
    """

    @dataclass
    class Context:
        """
        Specifies the aspects of an `ExperimentState` that should be taken into account for finding a suitable source or
        sink for handling data.

        Attributes:
            include_dataset_type:       True, if the type of the dataset should be taken into account, False otherwise
            include_prediction_scope:   True, if the scope of predictions should be taken into account, False otherwise
            include_fold:               True, if the cross validation fold should be taken into account, False otherwise
        """
        include_dataset_type: bool = True
        include_prediction_scope: bool = True
        include_fold: bool = True

    def __init__(self, context: Context = Context()):
        """
        :param context: A `Data.Context` to be used by default for finding a suitable source or sink this data can be
                        handled by
        """
        self.context = context


class FilePath:
    """
    The path to a file,  `Data` can be written to or read from.
    """

    def __init__(self, directory: str, file_name: str, suffix: str, context: Data.Context):
        """
        :param directory:   The path to the directory, where the file is located
        :param file_name:   The name of the file
        :param suffix:      The suffix of the file (with leading dot)
        :param context:     A `Data.Context` to be used to determine the path
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
