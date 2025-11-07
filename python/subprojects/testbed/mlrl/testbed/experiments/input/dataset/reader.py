"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow reading datasets from one or several sources.
"""
import logging as log
import sys

from dataclasses import replace
from typing import List, Optional, override

from mlrl.testbed.experiments.input.dataset.dataset import InputDataset
from mlrl.testbed.experiments.input.dataset.preprocessors import Preprocessor
from mlrl.testbed.experiments.input.reader import InputReader
from mlrl.testbed.experiments.input.sources import Source
from mlrl.testbed.experiments.state import ExperimentState


class DatasetReader(InputReader):
    """
    Allows reading a dataset from one or several sources.
    """

    def __init__(self, input_data: InputDataset, *sources: Source):
        """
        :param input_data:  The dataset that should be read
        :param sources:     The sources, the input dataset should be read from
        """
        super().__init__(input_data, *sources)
        self.preprocessors: List[Preprocessor] = []
        self.encoders: Optional[List[Preprocessor.Encoder]] = None

    def add_preprocessors(self, *preprocessors: Preprocessor):
        """
        Adds one or several preprocessors that should be applied to loaded datasets.

        :param preprocessors:   The preprocessors to be added
        :return:                The `DatasetReader` itself
        """
        if preprocessors:
            self.preprocessors.extend(preprocessors)
        return self

    def __read_all(self, state: ExperimentState) -> ExperimentState:
        for source in self.sources:
            try:
                new_state = replace(state)

                if source.read_from_source(new_state, self.input_data):
                    return new_state
            # pylint: disable=broad-exception-caught
            except Exception as error:
                log.error(str(error))

        log.error('Failed to load dataset!')
        sys.exit(1)

    @override
    def read(self, state: ExperimentState) -> ExperimentState:
        """
        See :func:`mlrl.testbed.experiments.input.reader.InputReader.read`
        """
        state = self.__read_all(state)
        encoders = self.encoders

        if encoders is None:
            encoders = [preprocessor.create_encoder() for preprocessor in self.preprocessors]
            self.encoders = encoders

        for encoder in encoders:
            state = replace(state, dataset=encoder.encode(state.dataset))

        return state
