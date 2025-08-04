"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing models for the calibration of probabilities that do not actually calibrate any
probabilities.
"""

from typing import Optional, override

from mlrl.testbed.experiments.output.data import TabularOutputData
from mlrl.testbed.experiments.table import Table

from mlrl.util.options import Options


class NoCalibrationModel(TabularOutputData):
    """
    Represents a model for the calibration of probabilities that does not actually calibrate any probabilities.
    """

    # pylint: disable=unused-argument
    @override
    def to_text(self, _: Options, **kwargs) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TextualOutputData.to_text`
        """
        return 'No calibration model used'

    # pylint: disable=unused-argument
    @override
    def to_table(self, _: Options, **kwargs) -> Optional[Table]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
        """
        return None
