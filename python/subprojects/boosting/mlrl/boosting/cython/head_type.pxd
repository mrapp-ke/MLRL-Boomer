cdef extern from "boosting/rule_evaluation/head_type_complete.hpp" namespace "boosting" nogil:

    cdef cppclass ICompleteHeadConfig:
        pass


cdef extern from "boosting/rule_evaluation/head_type_single.hpp" namespace "boosting" nogil:

    cdef cppclass ISingleLabelHeadConfig:
        pass


cdef class CompleteHeadConfig:

    # Attributes:

    cdef ICompleteHeadConfig* config_ptr


cdef class SingleLabelHeadConfig:

    # Attributes:

    cdef ISingleLabelHeadConfig* config_ptr
