from libcpp.cast cimport dynamic_cast
from libcpp.memory cimport unique_ptr


cdef extern from "common/output/label_space_info.hpp" nogil:

    cdef cppclass ILabelSpaceInfo:
        pass


cdef extern from "common/output/label_space_info_no.hpp" nogil:

    cdef cppclass INoLabelSpaceInfo(ILabelSpaceInfo):
        pass


cdef extern from "common/output/label_vector_set.hpp" nogil:

    cdef cppclass ILabelVectorSet(ILabelSpaceInfo):
        pass


ctypedef INoLabelSpaceInfo* NoLabelSpaceInfoPtr

ctypedef ILabelVectorSet* LabelVectorSetPtr


cdef class LabelSpaceInfo:

    # Functions:

    cdef ILabelSpaceInfo* get_label_space_info_ptr(self)


cdef class NoLabelSpaceInfo(LabelSpaceInfo):

    # Attributes:

    cdef unique_ptr[INoLabelSpaceInfo] label_space_info_ptr


cdef class LabelVectorSet(LabelSpaceInfo):

    # Attributes:

    cdef unique_ptr[ILabelVectorSet] label_vector_set_ptr


cdef inline LabelSpaceInfo create_label_space_info(unique_ptr[ILabelSpaceInfo] label_space_info_ptr):
    cdef ILabelSpaceInfo* ptr = label_space_info_ptr.release()
    cdef ILabelVectorSet* label_vector_set_ptr = dynamic_cast[LabelVectorSetPtr](ptr)
    cdef INoLabelSpaceInfo* no_label_space_info_ptr
    cdef LabelVectorSet label_vector_set
    cdef NoLabelSpaceInfo no_label_space_info

    if label_vector_set_ptr != NULL:
        label_vector_set = LabelVectorSet.__new__(LabelVectorSet)
        label_vector_set.label_vector_set_ptr = unique_ptr[ILabelVectorSet](label_vector_set_ptr)
        return label_vector_set
    else:
        no_label_space_info_ptr = dynamic_cast[NoLabelSpaceInfoPtr](ptr)

        if no_label_space_info_ptr != NULL:
            no_label_space_info = NoLabelSpaceInfo.__new__(NoLabelSpaceInfo)
            no_label_space_info.label_space_info_ptr = unique_ptr[INoLabelSpaceInfo](no_label_space_info_ptr)
            return no_label_space_info
        else:
            del ptr
            raise RuntimeError('Encountered unknown label space info type')
