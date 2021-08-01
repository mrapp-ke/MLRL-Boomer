from mlrl.common.cython._types cimport uint32
from mlrl.common.cython.input cimport NominalFeatureMask, INominalFeatureMask
from mlrl.common.cython.input cimport FeatureMatrix, IFeatureMatrix
from mlrl.common.cython.input cimport LabelMatrix, ILabelMatrix
from mlrl.common.cython.model cimport ModelBuilder, RuleModel, IModelBuilder, RuleModelImpl

from libcpp.memory cimport unique_ptr, shared_ptr


cdef extern from "common/rule_induction/rule_model_assemblage.hpp" nogil:

    cdef cppclass IRuleModelAssemblage:

        # Functions:

        unique_ptr[RuleModelImpl] induceRules(const INominalFeatureMask& nominalFeatureMask,
                                              const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix,
                                              uint32 randomState, IModelBuilder& modelBuilder)


    cdef cppclass IRuleModelAssemblageFactory:
        pass


cdef extern from "common/rule_induction/rule_model_assemblage_sequential.hpp" nogil:

    cdef cppclass SequentialRuleModelAssemblageFactoryImpl"SequentialRuleModelAssemblageFactory"(
            IRuleModelAssemblageFactory):
        pass


cdef class RuleModelAssemblage:

    # Attributes:

    cdef unique_ptr[IRuleModelAssemblage] rule_model_assemblage_ptr

    # Functions:

    cpdef RuleModel induce_rules(self, NominalFeatureMask nominal_feature_mask, FeatureMatrix feature_matrix,
                                 LabelMatrix label_matrix, uint32 random_state, ModelBuilder model_builder)


cdef class RuleModelAssemblageFactory:

    # Attributes:

    cdef shared_ptr[IRuleModelAssemblageFactory] rule_model_assemblage_factory_ptr


cdef class SequentialRuleModelAssemblageFactory(RuleModelAssemblageFactory):
    pass
