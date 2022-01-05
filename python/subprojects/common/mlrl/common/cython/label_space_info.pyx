"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""


cdef class LabelSpaceInfo:
    """
    A wrapper for the pure virtual C++ class `ILabelSpaceInfo`.
    """

    cdef ILabelSpaceInfo* get_label_space_info_ptr(self):
        pass


cdef class NoLabelSpaceInfo(LabelSpaceInfo):
    """
    A wrapper for the pure virtual C++ class `INoLabelSpaceInfo`.
    """

    cdef ILabelSpaceInfo* get_label_space_info_ptr(self):
        return self.label_space_info_ptr.get()


cdef class LabelVectorSet(LabelSpaceInfo):
    """
    A wrapper for the pure virtual C++ class `ILabelVectorSet`.
    """

    cdef ILabelSpaceInfo* get_label_space_info_ptr(self):
        return self.label_vector_set_ptr.get()
