cdef extern from "boosting/losses/loss_example_wise_logistic.hpp" namespace "boosting" nogil:

    cdef cppclass ExampleWiseLogisticLossConfigImpl"boosting::ExampleWiseLogisticLossConfig":
        pass


cdef extern from "boosting/losses/loss_label_wise_logistic.hpp" namespace "boosting" nogil:

    cdef cppclass LabelWiseLogisticLossConfigImpl"boosting::LabelWiseLogisticLossConfig":
        pass


cdef extern from "boosting/losses/loss_label_wise_squared_error.hpp" namespace "boosting" nogil:

    cdef cppclass LabelWiseSquaredErrorLossConfigImpl"boosting::LabelWiseSquaredErrorLossConfig":
        pass


cdef extern from "boosting/losses/loss_label_wise_squared_hinge.hpp" namespace "boosting" nogil:

    cdef cppclass LabelWiseSquaredHingeLossConfigImpl"boosting::LabelWiseSquaredHingeLossConfig":
        pass


cdef class ExampleWiseLogisticLossConfig:

    # Attributes:

    cdef ExampleWiseLogisticLossConfigImpl* config_ptr


cdef class LabelWiseLogisticLossConfig:

    # Attributes:

    cdef LabelWiseLogisticLossConfigImpl* config_ptr


cdef class LabelWiseSquaredErrorLossConfig:

    # Attributes:

    cdef LabelWiseSquaredErrorLossConfigImpl* config_ptr


cdef class LabelWiseSquaredHingeLossConfig:

    # Attributes:

    cdef LabelWiseSquaredHingeLossConfigImpl* config_ptr
