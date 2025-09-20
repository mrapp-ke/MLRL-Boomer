"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing output data to the log.
"""
import logging as log

from pathlib import Path
from typing import Callable, Optional, override

from mlrl.testbed.experiments.input.sources import Source
from mlrl.testbed.experiments.output.data import OutputData, TextualOutputData
from mlrl.testbed.experiments.output.sinks.sink import Sink
from mlrl.testbed.experiments.state import ExperimentState

from mlrl.util.options import Options


class LogSink(Sink):
    """
    Allows to write textual output data to the log.
    """

    SourceFactory = Callable[[Path], Source]

    def __init__(self, options: Options = Options(), source_factory: Optional[SourceFactory] = None):
        """
        :param options:         Options to be taken into account
        :param source_factory:  A factory that allows to create a `Source` that can read the data written to this sink
                                or None, if no such source is available
        """
        super().__init__(options=options)
        self._source_factory = source_factory

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

    @override
    def create_source(self, input_directory: Path) -> Optional[Source]:
        source_factory = self._source_factory
        return source_factory(input_directory) if source_factory else None
