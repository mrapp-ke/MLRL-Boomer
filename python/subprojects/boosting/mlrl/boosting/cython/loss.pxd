cdef extern from "boosting/losses/loss_example_wise_logistic.hpp" namespace "boosting" nogil:

    cdef cppclass IExampleWiseLogisticLossConfig:
        pass


cdef extern from "boosting/losses/loss_label_wise_logistic.hpp" namespace "boosting" nogil:

    cdef cppclass ILabelWiseLogisticLossConfig:
        pass


cdef extern from "boosting/losses/loss_label_wise_squared_error.hpp" namespace "boosting" nogil:

    cdef cppclass ILabelWiseSquaredErrorLossConfig:
        pass


cdef extern from "boosting/losses/loss_label_wise_squared_hinge.hpp" namespace "boosting" nogil:

    cdef cppclass ILabelWiseSquaredHingeLossConfig:
        pass


cdef class ExampleWiseLogisticLossConfig:

    # Attributes:

    cdef IExampleWiseLogisticLossConfig* config_ptr


cdef class LabelWiseLogisticLossConfig:

    # Attributes:

    cdef ILabelWiseLogisticLossConfig* config_ptr


cdef class LabelWiseSquaredErrorLossConfig:

    # Attributes:

    cdef ILabelWiseSquaredErrorLossConfig* config_ptr


cdef class LabelWiseSquaredHingeLossConfig:

    # Attributes:

    cdef ILabelWiseSquaredHingeLossConfig* config_ptr
