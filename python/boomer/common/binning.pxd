from boomer.common._arrays cimport uint32
from boomer.common.input_data cimport FeatureVector

cdef extern from "cpp/binning.h" nogil:

    cdef cppclass IBinningObserver:

        # Functions:

            onBinUpdate(uint32 binIndex, const Entry& entry)


    cdef cppclass IBinning:

        # Functions:

            void createBins(uint32 numBins, FeatureVector& featureVector, IBinningObserver& observer)


    cdef cppclass EqualFrequencyBinningImpl:

        # Functions:

            void createBins(uint32 numBins, FeatureVector& featureVector, IBinningObserver& observer)


    cdef cppclass EqualWidthBinningImpl:

        # Functions:

            void createBins(uint32 numBins, FeatureVector& featureVector, IBinningObserver& observer)