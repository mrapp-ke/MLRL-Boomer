from mlrl.common.cython._types cimport float32
from mlrl.common.cython.sampling cimport IInstanceSamplingFactory, InstanceSamplingFactory


cdef extern from "boosting/sampling/instance_sampling_with_replacement.hpp" namespace "boosting" nogil:

    cdef cppclass InstanceSamplingWithReplacementFactoryImpl"boosting::InstanceSamplingWithReplacementFactory"(
            IInstanceSamplingFactory):

        # Constructors:

        InstanceSamplingWithReplacementFactoryImpl(float32 sampleSize) except +


cdef extern from "boosting/sampling/instance_sampling_without_replacement.hpp" namespace "boosting" nogil:

    cdef cppclass InstanceSamplingWithoutReplacementFactoryImpl"boosting::InstanceSamplingWithoutReplacementFactory"(
            IInstanceSamplingFactory):

        # Constructors:

        InstanceSamplingWithoutReplacementFactoryImpl(float32 sampleSize) except +


cdef extern from "boosting/sampling/instance_sampling_stratified_label_wise.hpp" namespace "boosting" nogil:

    cdef cppclass LabelWiseStratifiedSamplingFactoryImpl"boosting::LabelWiseStratifiedSamplingFactory"(
            IInstanceSamplingFactory):

        # Constructors:

        LabelWiseStratifiedSamplingFactoryImpl(float32 sampleSize) except +


cdef extern from "boosting/sampling/instance_sampling_stratified_example_wise.hpp" namespace "boosting" nogil:

    cdef cppclass ExampleWiseStratifiedSamplingFactoryImpl"boosting::ExampleWiseStratifiedSamplingFactory"(
            IInstanceSamplingFactory):

        # Constructors:

        ExampleWiseStratifiedSamplingFactoryImpl(float32 sampleSize) except +


cdef extern from "boosting/sampling/instance_sampling_no.hpp" namespace "boosting" nogil:

    cdef cppclass NoInstanceSamplingFactoryImpl"boosting::NoInstanceSamplingFactory"(IInstanceSamplingFactory):
        pass


cdef class InstanceSamplingWithReplacementFactory(InstanceSamplingFactory):
    pass


cdef class InstanceSamplingWithoutReplacementFactory(InstanceSamplingFactory):
    pass


cdef class LabelWiseStratifiedSamplingFactory(InstanceSamplingFactory):
    pass


cdef class ExampleWiseStratifiedSamplingFactory(InstanceSamplingFactory):
    pass


cdef class NoInstanceSamplingFactory(InstanceSamplingFactory):
    pass
