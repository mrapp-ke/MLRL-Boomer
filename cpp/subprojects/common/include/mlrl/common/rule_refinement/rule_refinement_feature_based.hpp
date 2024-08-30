/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_vector.hpp"
#include "mlrl/common/rule_refinement/rule_refinement.hpp"

#include <memory>

/**
 * Allows to find the best refinements of existing rules, which result from adding a new condition that correspond to a
 * certain feature. The thresholds that may be used by the new condition result from the feature values of all training
 * examples for the respective feature.
 *
 * @tparam IndexVector The type of the vector that provides access to the indices of the outputs for which the refined
 *                     rule is allowed to predict
 */
template<typename IndexVector>
class FeatureBasedRuleRefinement final : public IRuleRefinement {
    private:

        const IndexVector& outputIndices_;

        const uint32 featureIndex_;

        const IWeightedStatistics& statistics_;

        const IFeatureVector& featureVector_;

        const uint32 numExamplesWithNonZeroWeights_;

    public:

        /**
         * @param outputIndices                 A reference to an object of template type `IndexVector` that provides
         *                                      access to the indices of the outputs for which the refined rule is
         *                                      allowed to predict
         * @param featureIndex                  The index of the feature, the new condition corresponds to
         * @param statistics                    A reference to an object of type `IWeightedStatistics` that should be
         *                                      used to search for potential refinements
         * @param featureVector                 A reference to an object of type `IFeatureVector` that should be used to
         *                                      search for potential refinements
         * @param numExamplesWithNonZeroWeights The total number of examples with non-zero weights that may be covered
         *                                      by a refinement
         */
        FeatureBasedRuleRefinement(const IndexVector& outputIndices, uint32 featureIndex,
                                   const IWeightedStatistics& statistics, const IFeatureVector& featureVector,
                                   uint32 numExamplesWithNonZeroWeights);

        void findRefinement(SingleRefinementComparator& comparator, uint32 minCoverage) const override;

        void findRefinement(FixedRefinementComparator& comparator, uint32 minCoverage) const override;
};
