"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing output data to the log.
"""
import logging as log

from typing import override

from mlrl.testbed.experiments.output.data import OutputData, TextualOutputData
from mlrl.testbed.experiments.output.sinks.sink import Sink
from mlrl.testbed.experiments.state import ExperimentState


class LogSink(Sink):
    """
    Allows to write textual output data to the log.
    """

    @override
    def write_to_sink(self, state: ExperimentState, output_data: OutputData, **kwargs):
        """
        See :func:`mlrl.testbed.experiments.output.sinks.sink.Sink.write_to_sink`
        """
        if isinstance(output_data, TextualOutputData):
            text = output_data.to_text(self.options, **kwargs)

            if text:
                context = output_data.get_context(type(self))
                title = TextualOutputData.Title(title=output_data.properties.name, context=context)
                log.info('%s:\n\n%s\n', title.format(state), text)
