from mlrl.common.cython.instance_sampling cimport IExampleWiseStratifiedInstanceSamplingConfig, \
    IOutputWiseStratifiedInstanceSamplingConfig
from mlrl.common.cython.partition_sampling cimport IExampleWiseStratifiedBiPartitionSamplingConfig, \
    IOutputWiseStratifiedBiPartitionSamplingConfig


cdef extern from "mlrl/common/learner_classification.hpp" nogil:

    cdef cppclass IOutputWiseStratifiedInstanceSamplingMixin"IRuleLearner::IOutputWiseStratifiedInstanceSamplingMixin":

        # Functions:

        IOutputWiseStratifiedInstanceSamplingConfig& useOutputWiseStratifiedInstanceSampling()


    cdef cppclass IExampleWiseStratifiedInstanceSamplingMixin \
        "IRuleLearner::IExampleWiseStratifiedInstanceSamplingMixin":

        # Functions:

        IExampleWiseStratifiedInstanceSamplingConfig& useExampleWiseStratifiedInstanceSampling()


    cdef cppclass IOutputWiseStratifiedBiPartitionSamplingMixin\
        "IRuleLearner::IOutputWiseStratifiedBiPartitionSamplingMixin":

        # Functions:

        IOutputWiseStratifiedBiPartitionSamplingConfig& useOutputWiseStratifiedBiPartitionSampling()


    cdef cppclass IExampleWiseStratifiedBiPartitionSamplingMixin\
        "IRuleLearner::IExampleWiseStratifiedBiPartitionSamplingMixin":

        # Functions:

        IExampleWiseStratifiedBiPartitionSamplingConfig& useExampleWiseStratifiedBiPartitionSampling()
