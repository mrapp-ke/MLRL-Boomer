from libcpp.cast cimport dynamic_cast
from libcpp.memory cimport unique_ptr

from mlrl.common.cython._types cimport uint32


cdef extern from "mlrl/common/prediction/output_space_info.hpp" nogil:

    cdef cppclass IOutputSpaceInfo:
        pass


cdef extern from "mlrl/common/prediction/output_space_info_no.hpp" nogil:

    cdef cppclass INoOutputSpaceInfo(IOutputSpaceInfo):
        pass


    unique_ptr[INoOutputSpaceInfo] createNoOutputSpaceInfo()


cdef extern from "mlrl/common/input/label_vector.hpp" nogil:

    cdef cppclass LabelVector:

        ctypedef const uint32* const_iterator

        ctypedef uint32* iterator

        # Constructors:

        LabelVector(uint32 numElements)

        # Functions:

        uint32 getNumElements() const

        iterator begin()

        const_iterator cbegin() const


ctypedef void (*LabelVectorVisitor)(const LabelVector&, uint32)


cdef extern from "mlrl/common/prediction/label_vector_set.hpp" nogil:

    cdef cppclass ILabelVectorSet(IOutputSpaceInfo):

        # Functions:

        void addLabelVector(unique_ptr[LabelVector] labelVectorPtr, uint32 frequency)

        void visit(LabelVectorVisitor) const


    unique_ptr[ILabelVectorSet] createLabelVectorSet()


ctypedef INoOutputSpaceInfo* NoOutputSpaceInfoPtr

ctypedef ILabelVectorSet* LabelVectorSetPtr


cdef extern from *:
    """
    #include "mlrl/common/prediction/label_vector_set.hpp"


    typedef void (*LabelVectorCythonVisitor)(void*, const LabelVector&, uint32);

    static inline LabelVectorSet::LabelVectorVisitor wrapLabelVectorVisitor(
            void* self, LabelVectorCythonVisitor visitor) {
        return [=](const LabelVector& labelVector, uint32 frequency) {
            visitor(self, labelVector, frequency);
        };
    }
    """

    ctypedef void (*LabelVectorCythonVisitor)(void*, const LabelVector&, uint32 frequency)

    LabelVectorVisitor wrapLabelVectorVisitor(void* self, LabelVectorCythonVisitor visitor)


cdef class OutputSpaceInfo:

    # Functions:

    cdef IOutputSpaceInfo* get_output_space_info_ptr(self)


cdef class NoOutputSpaceInfo(OutputSpaceInfo):

    # Attributes:

    cdef unique_ptr[INoOutputSpaceInfo] output_space_info_ptr


cdef class LabelVectorSet(OutputSpaceInfo):

    # Attributes:

    cdef unique_ptr[ILabelVectorSet] label_vector_set_ptr

    cdef object state

    cdef object visitor

    # Functions:

    cdef __visit_label_vector(self, const LabelVector& label_vector, uint32 frequency)

    cdef __serialize_label_vector(self, const LabelVector& label_vector, uint32 frequency)

    cdef unique_ptr[LabelVector] __deserialize_label_vector(self, object label_vector_state)


cdef inline OutputSpaceInfo create_output_space_info(unique_ptr[IOutputSpaceInfo] output_space_info_ptr):
    cdef IOutputSpaceInfo* ptr = output_space_info_ptr.release()
    cdef ILabelVectorSet* label_vector_set_ptr = dynamic_cast[LabelVectorSetPtr](ptr)
    cdef INoOutputSpaceInfo* no_output_space_info_ptr
    cdef LabelVectorSet label_vector_set
    cdef NoOutputSpaceInfo no_output_space_info

    if label_vector_set_ptr != NULL:
        label_vector_set = LabelVectorSet.__new__(LabelVectorSet)
        label_vector_set.label_vector_set_ptr = unique_ptr[ILabelVectorSet](label_vector_set_ptr)
        return label_vector_set
    else:
        no_output_space_info_ptr = dynamic_cast[NoOutputSpaceInfoPtr](ptr)

        if no_output_space_info_ptr != NULL:
            no_output_space_info = NoOutputSpaceInfo.__new__(NoOutputSpaceInfo)
            no_output_space_info.output_space_info_ptr = unique_ptr[INoOutputSpaceInfo](no_output_space_info_ptr)
            return no_output_space_info
        else:
            del ptr
            raise RuntimeError('Encountered unsupported IOutputSpaceInfo object')
