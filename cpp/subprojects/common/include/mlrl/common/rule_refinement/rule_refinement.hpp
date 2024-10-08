/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/indices/index_vector.hpp"
#include "mlrl/common/input/feature_matrix.hpp"
#include "mlrl/common/rule_refinement/feature_subspace.hpp"
#include "mlrl/common/rule_refinement/refinement_comparator_fixed.hpp"
#include "mlrl/common/rule_refinement/refinement_comparator_single.hpp"

#include <memory>

/**
 * Defines an interface for all classes that allow to find the best refinement of existing rules.
 */
class IRuleRefinement {
    public:

        virtual ~IRuleRefinement() {}

        /**
         * Finds the best refinement of an existing rule.
         *
         * @param comparator        A reference to an object of type `SingleRefinementComparator` that should be used
         *                          for comparing potential refinements
         * @param featureSubspace   A reference to an object of type `IFeatureSubspace` that should be used to search
         *                          for the potential refinements
         * @param featureIndices    A reference to an object of type `IIndexVector` that provides access to the indices
         *                          of the features that should be considered
         * @param outputIndices     A reference to an object of type `IIndexVector` that provides access to the indices
         *                          of the outputs for which the refinement(s) may predict
         * @param minCoverage       The minimum number of examples that must be covered by the refinements
         * @return                  True, if at least one refinement has been found, false otherwise
         */
        virtual bool findRefinement(SingleRefinementComparator& comparator, IFeatureSubspace& featureSubspace,
                                    const IIndexVector& featureIndices, const IIndexVector& outputIndices,
                                    uint32 minCoverage) const = 0;

        /**
         * Finds the best refinements of an existing rule.
         *
         * @param comparator        A reference to an object of type `FixedRefinementComparator` that should be used for
         *                          comparing potential refinements
         * @param featureSubspace   A reference to an object of type `IFeatureSubspace` that should be used to search
         *                          for the potential refinements
         * @param featureIndices    A reference to an object of type `IIndexVector` that provides access to the indices
         *                          of the features that should be considered
         * @param outputIndices     A reference to an object of type `IIndexVector` that provides access to the indices
         *                          of the outputs for which the refinement(s) may predict
         * @param minCoverage       The minimum number of examples that must be covered by the refinements
         * @return                  True, if at least one refinement has been found, false otherwise
         */
        virtual bool findRefinement(FixedRefinementComparator& comparator, IFeatureSubspace& featureSubspace,
                                    const IIndexVector& featureIndices, const IIndexVector& outputIndices,
                                    uint32 minCoverage) const = 0;
};

/**
 * Defines an interface for all factories that allow to create instances of the type `IRuleRefinement`.
 */
class IRuleRefinementFactory {
    public:

        virtual ~IRuleRefinementFactory() {}

        /**
         * Creates and returns a new object of type `IRuleRefinement`.
         *
         * @return An unique pointer to an object of type `IRuleRefinement` that has been created
         */
        virtual std::unique_ptr<IRuleRefinement> create() const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a method for finding the best refinements of existing
 * rules.
 */
class IRuleRefinementConfig {
    public:

        virtual ~IRuleRefinementConfig() {}

        /**
         * Creates and returns an new object of type `IRuleRefinementFactory`.
         *
         * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the feature
         *                      values of the training examples
         * @param numOutputs    The total number of available outputs
         * @return              An unique pointer to an object of type `IRuleRefinementFactory` that has been created
         */
        virtual std::unique_ptr<IRuleRefinementFactory> createRuleRefinementFactory(const IFeatureMatrix& featureMatrix,
                                                                                    uint32 numOutputs) const = 0;
};
