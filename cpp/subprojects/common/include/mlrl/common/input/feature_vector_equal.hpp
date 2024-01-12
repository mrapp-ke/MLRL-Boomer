/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_vector.hpp"

/**
 * A feature vector that does not actually store any values. It is used in cases where all training examples have the
 * same value for a certain feature.
 */
class EqualFeatureVector final : public IFeatureVector {
    public:

        void searchForRefinement(RuleRefinementSearch& ruleRefinementSearch, SingleRefinementComparator& comparator,
                                 uint32 minCoverage) const override;

        void searchForRefinement(RuleRefinementSearch& ruleRefinementSearch, FixedRefinementComparator& comparator,
                                 uint32 minCoverage) const override;

        void updateCoverageMaskAndStatistics(const Interval& interval, CoverageMask& coverageMask,
                                             uint32 indicatorValue, IWeightedStatistics& statistics) const override;

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const Interval& interval) const override;

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const CoverageMask& coverageMask) const override;
};
