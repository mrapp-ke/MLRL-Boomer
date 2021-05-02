from common.cython._types cimport float32
from common.cython.sampling cimport IInstanceSubSampling


cdef extern from "boosting/sampling/instance_sampling_gradient_based_labelset.hpp" nogil:

    cdef cppclass GradientBasedLabelSetImpl"boosting::GradientBasedLabelSet"(IInstanceSubSampling):

        # Constructors:

        GradientBasedLabelSetImpl(float32 sampleSize) except +


cdef extern from "boosting/sampling/instance_sampling_gradient_based_labelwise.hpp" nogil:

    cdef cppclass GradientBasedLabelWiseImpl"boosting::GradientBasedLabelWise"(IInstanceSubSampling):

        # Constructors:

        GradientBasedLabelWiseImpl(float32 sampleSize) except +


cdef extern from "boosting/sampling/instance_sampling_iterative_stratification_labelwise.hpp" nogil:

    cdef cppclass IterativeStratificationLabelWiseImpl"boosting::IterativeStratificationLabelWise"(IInstanceSubSampling):

        # Constructors:

        IterativeStratificationLabelWiseImpl(float32 sampleSize) except +


cdef extern from "boosting/sampling/instance_sampling_iterative_stratification_labelset.hpp" nogil:

    cdef cppclass IterativeStratificationLabelSetImpl"boosting::IterativeStratificationLabelSet"(IInstanceSubSampling):

        # Constructors:

        IterativeStratificationLabelSetImpl(float32 sampleSize) except +