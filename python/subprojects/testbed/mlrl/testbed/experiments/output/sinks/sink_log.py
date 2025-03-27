"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing output data to the log.
"""
import logging as log

from mlrl.common.config.options import Options

from mlrl.testbed.experiments.output.sinks.sink import Sink
from mlrl.testbed.experiments.state import ExperimentState


class LogSink(Sink):
    """
    Allows to write output data to the log.
    """

    class TitleFormatter:
        """
        Allows to format the title that is printed before the output data.
        """

        def __init__(self,
                     title: str,
                     formatter_options: ExperimentState.FormatterOptions = ExperimentState.FormatterOptions()):
            """
            :param title:               A title
            :param formatter_options:   The options to be used by the formatter
            """
            self.title = title
            self.formatter_options = formatter_options

        def __format_dataset_type(self, state: ExperimentState) -> str:
            if self.formatter_options.include_dataset_type:
                return ' for ' + state.dataset.type.value + ' data'
            return ''

        def __format_fold(self, state: ExperimentState) -> str:
            if self.formatter_options.include_fold:
                fold = state.fold

                if fold.is_cross_validation_used:
                    if fold.index is None:
                        formatted_fold = 'Average across ' + str(fold.num_folds) + ' folds'
                    else:
                        formatted_fold = 'Fold ' + str(fold.index + 1)

                    return ' (' + formatted_fold + ')'

            return ''

        def __format_prediction_scope(self, state: ExperimentState) -> str:
            if self.formatter_options.include_prediction_scope:
                prediction_result = state.prediction_result

                if prediction_result:
                    prediction_scope = prediction_result.prediction_scope

                    if not prediction_scope.is_global:
                        return ' using a model of size ' + str(prediction_scope.model_size)

            return ''

        def format(self, state: ExperimentState) -> str:
            """
            Formats and returns the title that is printed before the output data.

            :param state: The state from which the output data is generated
            """
            return self.title + self.__format_dataset_type(state) + self.__format_prediction_scope(
                state) + self.__format_fold(state)

    def __init__(self, title_formatter: TitleFormatter, options: Options = Options()):
        """
        :param title_formatter: A `TitleFormatter` to be used for formatting the title that is printed before the
                                output data
        """
        super().__init__(options)
        self.title_formatter = title_formatter

    def write_to_sink(self, state: ExperimentState, output_data, **kwargs):
        """
        See :func:`mlrl.testbed.experiments.output.sinks.sink.Sink.write_to_sink`
        """
        text = output_data.to_text(self.options, **kwargs)

        if text:
            log.info('%s:\n\n%s\n', self.title_formatter.format(state), text)
