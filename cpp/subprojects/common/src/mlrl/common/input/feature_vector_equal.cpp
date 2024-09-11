#include "mlrl/common/input/feature_vector_equal.hpp"

void EqualFeatureVector::searchForRefinement(FeatureBasedSearch& featureBasedSearch,
                                             SingleRefinementComparator& comparator,
                                             const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
                                             uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                             Refinement& refinement) const {}

void EqualFeatureVector::searchForRefinement(FeatureBasedSearch& featureBasedSearch,
                                             FixedRefinementComparator& comparator,
                                             const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
                                             uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                             Refinement& refinement) const {}

std::unique_ptr<IResettableStatisticsSubset> EqualFeatureVector::createStatisticsSubset(
  const IWeightedStatistics& statistics, const CompleteIndexVector& outputIndices) const {
    throw std::runtime_error("Function EqualFeatureVector::createStatisticsSubset should never be called");
}

std::unique_ptr<IResettableStatisticsSubset> EqualFeatureVector::createStatisticsSubset(
  const IWeightedStatistics& statistics, const PartialIndexVector& outputIndices) const {
    throw std::runtime_error("Function EqualFeatureVector::createStatisticsSubset should never be called");
}

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
