"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.cast cimport dynamic_cast

from cython.operator cimport dereference

from mlrl.common.cython._serialization cimport check_pickled_state, image_to_bytes

from abc import abstractmethod

import numpy as np


cdef class OutputSpaceInfo:
    """
    Provides information about the output space that may be used as a basis for making predictions.
    """

    cdef IOutputSpaceInfo* get_output_space_info_ptr(self):
        pass


cdef class NoOutputSpaceInfo(OutputSpaceInfo):
    """
    Does not provide any information about the output space.
    """

    cdef IOutputSpaceInfo* get_output_space_info_ptr(self):
        return self.output_space_info_ptr.get()

    def __reduce__(self):
        return (NoOutputSpaceInfo, (), ())

    def __setstate__(self, state):
        self.output_space_info_ptr = createNoOutputSpaceInfo()


class LabelVectorSetVisitor:
    """
    Defines the methods that must be implemented by a visitor that accesses the label vectors and frequencies stored by
    a `LabelVectorSet`.
    """

    @abstractmethod
    def visit_label_vector(self, label_vector: np.ndarray, frequency: int):
        """
        Must be implemented by subclasses in order to visit label vectors and their frequencies.

        :param label_vector:    An `np.ndarray` of type `uint8`, shape `(num_relevant_labels)` that stores the indices
                                of the relevant labels
        :param frequency:       The frequency of the label vector
        """


cdef class LabelVectorSet(OutputSpaceInfo):
    """
    Stores a set of unique label vectors, as well as their frequency.
    """

    def visit(self, visitor: LabelVectorSetVisitor):
        """
        Visits the label vectors and frequencies stored by the set.

        :param visitor: The `LabelVectorSetVisitor` that should be used to access the label vectors and frequencies
        """
        self.visitor = visitor
        self.label_vector_set_ptr.get().visit(
            wrapLabelVectorVisitor(<void*>self, <LabelVectorCythonVisitor>self.__visit_label_vector))
        self.visitor = None

    cdef IOutputSpaceInfo* get_output_space_info_ptr(self):
        return self.label_vector_set_ptr.get()

    cdef __visit_label_vector(self, const LabelVector& label_vector, uint32 frequency):
        cdef uint32 num_elements = label_vector.getNumElements()
        label_indices = np.asarray(<uint32[:num_elements]>label_vector.cbegin() if num_elements > 0 else [])
        self.visitor.visit_label_vector(label_indices, frequency)

    def __reduce__(self):
        cdef bytes state = image_to_bytes(serializeOutputSpaceInfo(dereference(self.label_vector_set_ptr.get())))
        return (LabelVectorSet, (), state)

    def __setstate__(self, state):
        check_pickled_state(state, 'LabelVectorSet')
        cdef const uint8[::1] data = state
        cdef unique_ptr[IOutputSpaceInfo] output_space_info_ptr = deserializeOutputSpaceInfo(
            &data[0] if data.shape[0] > 0 else NULL, data.shape[0])
        cdef ILabelVectorSet* label_vector_set_ptr = dynamic_cast[LabelVectorSetPtr](output_space_info_ptr.get())

        if label_vector_set_ptr == NULL:
            raise ValueError('The pickled data does not contain an object of type LabelVectorSet')

        output_space_info_ptr.release()
        self.label_vector_set_ptr = unique_ptr[ILabelVectorSet](label_vector_set_ptr)
