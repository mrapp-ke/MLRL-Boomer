from boomer.common._arrays cimport uint32
from boomer.common.input_data cimport IFeatureMatrix, INominalFeatureVector
from boomer.common.rule_refinement cimport IRuleRefinement
from boomer.common.statistics cimport AbstractStatistics
from boomer.common.sub_sampling cimport IWeightVector

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/thresholds.h" nogil:

    cdef cppclass IThresholdsSubset:

        # Functions:

        IRuleRefinement* createRuleRefinement(uint32 featureIndex)


    cdef cppclass AbstractThresholds:

        # Functions:

        uint32 getNumRows()

        uint32 getNumCols()

        IThresholdsSubset* createSubset(IWeightVector* weights)


    cdef cppclass ExactThresholdsImpl(AbstractThresholds):

        # Constructors:

        ExactThresholdsImpl(shared_ptr[IFeatureMatrix] featureMatrixPtr,
                            shared_ptr[INominalFeatureVector] nominalFeatureVectorPtr,
                            shared_ptr[AbstractStatistics] statisticsPtr) except +
