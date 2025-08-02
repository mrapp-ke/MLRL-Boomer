"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow reading datasets from a source.
"""
from dataclasses import replace
from typing import List, Optional, override

from mlrl.testbed.experiments.input.dataset.dataset import InputDataset
from mlrl.testbed.experiments.input.dataset.preprocessors import Preprocessor
from mlrl.testbed.experiments.input.reader import InputReader
from mlrl.testbed.experiments.input.sources import Source
from mlrl.testbed.experiments.state import ExperimentState


class DatasetReader(InputReader):
    """
    Allows reading a dataset from a source.
    """

    def __init__(self, source: Source, input_data: InputDataset):
        """
        :param source:      The source, the input dataset should be read from
        :param input_data:  The dataset that should be read
        """
        super().__init__(source=source, input_data=input_data)
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

    @override
    def read(self, state: ExperimentState) -> ExperimentState:
        """
        See :func:`mlrl.testbed.experiments.input.reader.InputReader.read`
        """
        state = super().read(state)
        encoders = self.encoders

        if encoders is None:
            encoders = [preprocessor.create_encoder() for preprocessor in self.preprocessors]
            self.encoders = encoders

        for encoder in encoders:
            state = replace(state, dataset=encoder.encode(state.dataset))

        return state
