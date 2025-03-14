"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.utility cimport move


cdef class ExampleWeights:
    """
    Provides access to the weights of individual training examples.
    """

    cdef IExampleWeights* get_example_weights_ptr(self):
        pass


cdef class EqualExampleWeights(ExampleWeights):
    """
    Provides access to the weights of individual training examples in cases where all examples have equal weights.
    """
    
    def __cinit__(self, uint32 num_examples):
        """
        :param num_elements: The total number of available examples
        """
        self.example_weights_ptr = createEqualExampleWeights(num_examples)

    cdef IExampleWeights* get_example_weights_ptr(self):
        return self.example_weights_ptr.get()


cdef class RealValuedExampleWeights(ExampleWeights):
    """
    Provides access to the weights of individual training examples in cases where the examples have real-valued weights.
    """
    
    def __cinit__(self, const float32[::1] example_weights not None):
        """
        :param num_elements: The total number of available examples
        """
        cdef uint32 num_examples = example_weights.shape[0]
        cdef unique_ptr[IRealValuedExampleWeights] example_weights_ptr = createRealValuedExampleWeights(num_examples)
        cdef uint32 i

        for i in range(num_examples):
            example_weights_ptr.get().setWeight(i, example_weights[i])

        self.example_weights_ptr = move(example_weights_ptr)


    cdef IExampleWeights* get_example_weights_ptr(self):
        return self.example_weights_ptr.get()
