from mlrl.common.cython._types cimport float32


cdef extern from "common/sampling/instance_sampling_stratified_example_wise.hpp" nogil:

    cdef cppclass ExampleWiseStratifiedInstanceSamplingConfigImpl"ExampleWiseStratifiedInstanceSamplingConfig":

        # Functions:

        float32 getSampleSize() const

        ExampleWiseStratifiedInstanceSamplingConfigImpl& setSampleSize(float32 sampleSize)


cdef extern from "common/sampling/instance_sampling_stratified_label_wise.hpp" nogil:

    cdef cppclass LabelWiseStratifiedInstanceSamplingConfigImpl"LabelWiseStratifiedInstanceSamplingConfig":

        # Functions:

        float32 getSampleSize() const

        LabelWiseStratifiedInstanceSamplingConfigImpl& setSampleSize(float32 sampleSize)


cdef extern from "common/sampling/instance_sampling_with_replacement.hpp" nogil:

    cdef cppclass InstanceSamplingWithReplacementConfigImpl"InstanceSamplingWithReplacementConfig":

        # Functions:

        float32 getSampleSize() const

        InstanceSamplingWithReplacementConfigImpl& setSampleSize(float32 sampleSize)


cdef extern from "common/sampling/instance_sampling_without_replacement.hpp" nogil:

    cdef cppclass InstanceSamplingWithoutReplacementConfigImpl"InstanceSamplingWithoutReplacementConfig":

        # Functions:

        float32 getSampleSize() const

        InstanceSamplingWithoutReplacementConfigImpl& setSampleSize(float32 sampleSize)


cdef class ExampleWiseStratifiedInstanceSamplingConfig:

    # Attributes:

    cdef ExampleWiseStratifiedInstanceSamplingConfigImpl* config_ptr


cdef class LabelWiseStratifiedInstanceSamplingConfig:

    # Attributes:

    cdef LabelWiseStratifiedInstanceSamplingConfigImpl* config_ptr


cdef class InstanceSamplingWithReplacementConfig:

    # Attributes:

    cdef InstanceSamplingWithReplacementConfigImpl* config_ptr


cdef class InstanceSamplingWithoutReplacementConfig:

    # Attributes:

    cdef InstanceSamplingWithoutReplacementConfigImpl* config_ptr
