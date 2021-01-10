"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.utility cimport move


cdef class RuleModel:
    """
    A wrapper for the C++ class `RuleModel`.
    """
    pass


cdef class ModelBuilder:
    """
    A wrapper for the pure virtual C++ class `IModelBuilder`.
    """

    cdef RuleModel build(self):
        """
        Builds and returns the model.

        :return: The model that has been built
        """
        cdef RuleModel model = RuleModel()
        model.model_ptr = move(self.model_builder_ptr.get().build())
        return model
