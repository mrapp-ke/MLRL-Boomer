"""
@author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.memory cimport make_unique


cdef class FeatureBinningFactory:
    """
    A wrapper for the pure virtual C++ class `IFeatureBinningFactory`.
    """
    pass


cdef class EqualFrequencyFeatureBinningFactory(FeatureBinningFactory):
    """
    A wrapper for the C++ class `EqualFrequencyFeatureBinningFactory`.
    """

    def __cinit__(self, float32 bin_ratio, uint32 min_bins, uint32 max_bins):
        """
        :param bin_ratio:   A percentage that specifies how many bins should be used
        :param min_bins:    The minimum number of bins to be used
        :param max_bins:    The maximum number of bins to be used
        """
        self.feature_binning_factory_ptr = <unique_ptr[IFeatureBinningFactory]>make_unique[EqualFrequencyFeatureBinningFactoryImpl](
            bin_ratio, min_bins, max_bins)


cdef class EqualWidthFeatureBinningFactory(FeatureBinningFactory):
    """
    A wrapper for the C++ class `EqualWidthFeatureBinningFactory`.
    """

    def __cinit__(self, float32 bin_ratio, uint32 min_bins, uint32 max_bins):
        """
        :param bin_ratio:   A percentage that specifies how many bins should be used
        :param min_bins:    The minimum number of bins to be used
        :param max_bins:    The maximum number of bins to be used
        """
        self.feature_binning_factory_ptr = <unique_ptr[IFeatureBinningFactory]>make_unique[EqualWidthFeatureBinningFactoryImpl](
            bin_ratio, min_bins, max_bins)
