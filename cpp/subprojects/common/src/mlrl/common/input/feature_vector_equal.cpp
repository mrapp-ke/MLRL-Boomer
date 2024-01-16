#include "mlrl/common/input/feature_vector_equal.hpp"

void EqualFeatureVector::searchForRefinement(RuleRefinementSearch& ruleRefinementSearch,
                                             IWeightedStatisticsSubset& statistics,
                                             SingleRefinementComparator& comparator, uint32 minCoverage,
                                             Refinement& refinement) const {}

void EqualFeatureVector::searchForRefinement(RuleRefinementSearch& ruleRefinementSearch,
                                             IWeightedStatisticsSubset& statistics,
                                             FixedRefinementComparator& comparator, uint32 minCoverage,
                                             Refinement& refinement) const {}

void EqualFeatureVector::updateCoverageMaskAndStatistics(const Interval& interval, CoverageMask& CoverageMask,
                                                         uint32 indicatorValue, IWeightedStatistics& statistics) const {
    throw std::runtime_error("Function EqualFeatureVector::updateCoverageMaskAndStatistics should never be called");
}

std::unique_ptr<IFeatureVector> EqualFeatureVector::createFilteredFeatureVector(
  std::unique_ptr<IFeatureVector>& existing, const Interval& interval) const {
    throw std::runtime_error("Function EqualFeatureVector::createFilteredFeatureVector should never be called");
}

std::unique_ptr<IFeatureVector> EqualFeatureVector::createFilteredFeatureVector(
  std::unique_ptr<IFeatureVector>& existing, const CoverageMask& coverageMask) const {
    return std::make_unique<EqualFeatureVector>();
}
