cdef extern from "seco/rule_evaluation/head_type_partial.hpp" namespace "seco" nogil:

    cdef cppclass IPartialHeadConfig:
        pass


cdef extern from "seco/rule_evaluation/head_type_single.hpp" namespace "seco" nogil:

    cdef cppclass ISingleLabelHeadConfig:
        pass


cdef class PartialHeadConfig:

    # Attributes:

    cdef IPartialHeadConfig* config_ptr


cdef class SingleLabelHeadConfig:

    # Attributes:

    cdef ISingleLabelHeadConfig* config_ptr
