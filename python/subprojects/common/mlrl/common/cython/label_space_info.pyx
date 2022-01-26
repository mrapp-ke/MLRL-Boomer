"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""


cdef class LabelSpaceInfo:
    """
    Provides information about the label space that may be used as a basis for making predictions.
    """

    cdef ILabelSpaceInfo* get_label_space_info_ptr(self):
        pass


cdef class NoLabelSpaceInfo(LabelSpaceInfo):
    """
    Does not provide any information about the label space.
    """

    cdef ILabelSpaceInfo* get_label_space_info_ptr(self):
        return self.label_space_info_ptr.get()


cdef class LabelVectorSet(LabelSpaceInfo):
    """
    Stores a set of unique label vectors, as well as their frequency.
    """

    cdef ILabelSpaceInfo* get_label_space_info_ptr(self):
        return self.label_vector_set_ptr.get()
