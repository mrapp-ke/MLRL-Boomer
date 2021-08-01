"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from cython.operator cimport dereference

from libcpp.memory cimport make_shared
from libcpp.utility cimport move


cdef class RuleModelAssemblage:
    """
    A wrapper for the pure virtual C++ class `IRuleModelAssemblage`.
    """

    cpdef RuleModel induce_rules(self, NominalFeatureMask nominal_feature_mask, FeatureMatrix feature_matrix,
                                 LabelMatrix label_matrix, uint32 random_state, ModelBuilder model_builder):
        cdef unique_ptr[RuleModelImpl] rule_model_ptr = self.rule_model_assemblage_ptr.get().induceRules(
            dereference(nominal_feature_mask.nominal_feature_mask_ptr), dereference(feature_matrix.feature_matrix_ptr),
            dereference(label_matrix.label_matrix_ptr), random_state,
            dereference(model_builder.model_builder_ptr))
        cdef RuleModel model = RuleModel.__new__(RuleModel)
        model.model_ptr = move(rule_model_ptr)
        return model


cdef class SequentialRuleModelAssemblageFactory(RuleModelAssemblageFactory):
    """
    A wrapper for the C++ class `SequentialRuleModelAssemblageFactory`.
    """

    def __cinit__(self):
        self.rule_model_assemblage_factory_ptr = <shared_ptr[IRuleModelAssemblageFactory]>make_shared[SequentialRuleModelAssemblageFactoryImpl]()
