"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython.validation import assert_greater_or_equal


cdef class ManualMultiThreadingConfig:
    """
    Allows to configure the multi-threading behavior of a parallelizable algorithm by manually specifying the number of
    threads to be used.
    """

    def get_num_preferred_threads(self) -> int:
        """
        Returns the number of preferred threads.

        :return: The number of preferred threads or 0, if all available CPU cores are utilized
        """
        return self.config_ptr.getNumPreferredThreads()

    def set_num_preferred_threads(self, num_preferred_threads: int) -> ManualMultiThreadingConfig:
        """
        Sets the number of preferred threads. If not enough CPU cores are available or if multi-threading support was
        disabled at compile-time, as many threads as possible will be used.

        :param num_preferred_threads:   The preferred number of threads. Must be at least 1 or 0, if all available CPU
                                        cores should be utilized
        :return:                        A `ManualMultiThreadingConfig` that allows further configuration of the
                                        multi-threading behavior
        """
        if num_preferred_threads != 0:
            assert_greater_or_equal('num_preferred_threads', num_preferred_threads, 1)
        self.config_ptr.setNumPreferredThreads(num_preferred_threads)
        return self
