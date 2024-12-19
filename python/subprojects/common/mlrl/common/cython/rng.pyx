"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython.validation import assert_greater, assert_greater_or_equal, assert_less


cdef class RNGConfig:
    """
    Allows to configure random number generators.
    """

    def get_random_state(self) -> int:
        """
        Returns the seed that is used by random number generators.

        :return: The seed that is used
        """
        return self.config_ptr.getRandomState()

    def set_random_state(self, random_state: int) -> RNGConfig:
        """
        Sets the seed that should be used by random number generators.
        
        :param random_state:    The seed that should be used. Must be at least 1
        :return:                An `RNGConfig` that allows further configuration of the random number generators
        """
        assert_greater_or_equal('random_state', random_state, 1)
        self.config_ptr.setRandomState(random_state)
        return self
