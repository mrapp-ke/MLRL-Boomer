"""
@author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython.feature_binning cimport FeatureBinningFactory

from libcpp.utility cimport move
from libcpp.memory cimport unique_ptr, make_unique


cdef class ApproximateThresholdsFactory(ThresholdsFactory):
    """
    A wrapper for the C++ class `ApproximateThresholdsFactory`.
    """

    def __cinit__(self, FeatureBinningFactory feature_binning_factory not None, uint32 num_threads):
        """
        :param feature_binning_factory: The `FeatureBinningFactory` to be used
        :param num_threads:             The number of CPU threads to be used to update statistics in parallel. Must be
                                        at least 1
        """
        self.thresholds_factory_ptr = <unique_ptr[IThresholdsFactory]>make_unique[ApproximateThresholdsFactoryImpl](
            move(feature_binning_factory.feature_binning_factory_ptr), num_threads)
