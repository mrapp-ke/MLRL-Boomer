"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing output data to the log.
"""
import logging as log

from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.sinks.sink import Sink
from mlrl.testbed.experiments.state import ExperimentState


class LogSink(Sink):
    """
    Allows to write textual output data to the log.
    """

    class Title:
        """
        A title that is printed before output data.
        """

        def __init__(self, title: str, context: ExperimentState.Context):
            """
            :param title:   A title
            :param context: An `ExperimentState.Context` to be used for formatting the title
            """
            self.title = title
            self.context = context

        def __format_dataset_type(self, state: ExperimentState) -> str:
            if self.context.include_dataset_type:
                return ' for ' + state.dataset.type.value + ' data'
            return ''

        def __format_fold(self, state: ExperimentState) -> str:
            if self.context.include_fold:
                fold = state.fold

                if fold.is_cross_validation_used:
                    if fold.index is None:
                        formatted_fold = 'Average across ' + str(fold.num_folds) + ' folds'
                    else:
                        formatted_fold = 'Fold ' + str(fold.index + 1)

                    return ' (' + formatted_fold + ')'

            return ''

        def __format_prediction_scope(self, state: ExperimentState) -> str:
            if self.context.include_prediction_scope:
                prediction_result = state.prediction_result

                if prediction_result:
                    prediction_scope = prediction_result.prediction_scope

                    if not prediction_scope.is_global:
                        return ' using a model of size ' + str(prediction_scope.model_size)

            return ''

        def format(self, state: ExperimentState) -> str:
            """
            Formats and returns the title that is printed before the output data.

            :param state: The state from which the output data has been generated
            """
            return self.title + self.__format_dataset_type(state) + self.__format_prediction_scope(
                state) + self.__format_fold(state)

    def write_to_sink(self, state: ExperimentState, output_data: OutputData, **kwargs):
        """
        See :func:`mlrl.testbed.experiments.output.sinks.sink.Sink.write_to_sink`
        """
        text = output_data.to_text(self.options, **kwargs)

        if text:
            context = output_data.get_context(type(self))
            title = LogSink.Title(title=output_data.name, context=context)
            log.info('%s:\n\n%s\n', title.format(state), text)
