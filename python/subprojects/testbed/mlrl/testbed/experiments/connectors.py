"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for passing data to connectors that interact with the environment this software runs in.
"""
from abc import ABC, abstractmethod
from typing import Optional

from mlrl.testbed.experiments.data import Data
from mlrl.testbed.experiments.state import ExperimentState


class Connector(ABC):
    """
    An abstract base class for all connectors that interact with the environment this software runs in.
    """

    class Session(ABC):
        """
        An abstract base class for all sessions of a `Connector that can be used for exchanging data with the
        environment this software runs in.
        """

        @abstractmethod
        def exchange(self) -> Optional[Data]:
            """
            Exchanges data with the environment.

            :return: Data that has been received from the environment or None, if no data has been received
            """

    @abstractmethod
    def open_session(self, state: ExperimentState) -> Session:
        """
        Opens a session with for interchanging data with the environment.

        :param state:   The current state of the experiment
        :return:        The session that has been opened
        """
