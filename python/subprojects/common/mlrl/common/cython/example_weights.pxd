from libcpp.memory cimport unique_ptr

from mlrl.common.cython._types cimport float32, uint32


cdef extern from "mlrl/common/input/example_weights.hpp" nogil:

    cdef cppclass IExampleWeights:
        pass


cdef extern from "mlrl/common/input/example_weights_equal.hpp" nogil:

    cdef cppclass IEqualExampleWeights(IExampleWeights):
        pass

    unique_ptr[IEqualExampleWeights] createEqualExampleWeights(uint32 numExamples)


cdef extern from "mlrl/common/input/example_weights_real_valued.hpp" nogil:

    cdef cppclass IRealValuedExampleWeights(IExampleWeights):
        
        # Functions

        void setWeight(uint32 index, float32 weight)

    unique_ptr[IRealValuedExampleWeights] createRealValuedExampleWeights(uint32 numExamples)


cdef class ExampleWeights:

    # Functions:

    cdef IExampleWeights* get_example_weights_ptr(self)


cdef class EqualExampleWeights(ExampleWeights):
    
    # Attributes:

    cdef unique_ptr[IEqualExampleWeights] example_weights_ptr


cdef class RealValuedExampleWeights(ExampleWeights):
    
    # Attributes:

    cdef unique_ptr[IRealValuedExampleWeights] example_weights_ptr
