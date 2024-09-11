#include "mlrl/common/rule_refinement/feature_based_search.hpp"

#include "feature_based_search_binary.hpp"
#include "feature_based_search_binned.hpp"
#include "feature_based_search_nominal.hpp"
#include "feature_based_search_numerical.hpp"
#include "feature_based_search_ordinal.hpp"

static inline std::unique_ptr<IResettableStatisticsSubset> createStatisticsSubset(
  const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
  const MissingFeatureVector& missingFeatureVector) {
    std::unique_ptr<IResettableStatisticsSubset> statisticsSubsetPtr;
    auto partialIndexVectorVisitor = [&](const PartialIndexVector& partialIndexVector) {
        statisticsSubsetPtr = statistics.createSubset(missingFeatureVector, partialIndexVector);
    };
    auto completeIndexVectorVisitor = [&](const CompleteIndexVector& completeIndexVector) {
        statisticsSubsetPtr = statistics.createSubset(missingFeatureVector, completeIndexVector);
    };
    outputIndices.visit(partialIndexVectorVisitor, completeIndexVectorVisitor);
    return statisticsSubsetPtr;
}

void FeatureBasedSearch::searchForNumericalRefinement(
  const NumericalFeatureVector& featureVector, const MissingFeatureVector& missingFeatureVector,
  SingleRefinementComparator& comparator, const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
  uint32 numExamplesWithNonZeroWeights, uint32 minCoverage, Refinement& refinement) const {
    std::unique_ptr<IResettableStatisticsSubset> statisticsSubsetPtr =
      createStatisticsSubset(statistics, outputIndices, missingFeatureVector);
    searchForNumericalRefinementInternally(featureVector, *statisticsSubsetPtr, comparator,
                                           numExamplesWithNonZeroWeights, minCoverage, refinement);
}

void FeatureBasedSearch::searchForNumericalRefinement(
  const NumericalFeatureVector& featureVector, const MissingFeatureVector& missingFeatureVector,
  FixedRefinementComparator& comparator, const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
  uint32 numExamplesWithNonZeroWeights, uint32 minCoverage, Refinement& refinement) const {
    std::unique_ptr<IResettableStatisticsSubset> statisticsSubsetPtr =
      createStatisticsSubset(statistics, outputIndices, missingFeatureVector);
    searchForNumericalRefinementInternally(featureVector, *statisticsSubsetPtr, comparator,
                                           numExamplesWithNonZeroWeights, minCoverage, refinement);
}

void FeatureBasedSearch::searchForNominalRefinement(
  const NominalFeatureVector& featureVector, const MissingFeatureVector& missingFeatureVector,
  SingleRefinementComparator& comparator, const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
  uint32 numExamplesWithNonZeroWeights, uint32 minCoverage, Refinement& refinement) const {
    std::unique_ptr<IResettableStatisticsSubset> statisticsSubsetPtr =
      createStatisticsSubset(statistics, outputIndices, missingFeatureVector);
    searchForNominalRefinementInternally(featureVector, *statisticsSubsetPtr, comparator, numExamplesWithNonZeroWeights,
                                         minCoverage, refinement);
}

void FeatureBasedSearch::searchForNominalRefinement(
  const NominalFeatureVector& featureVector, const MissingFeatureVector& missingFeatureVector,
  FixedRefinementComparator& comparator, const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
  uint32 numExamplesWithNonZeroWeights, uint32 minCoverage, Refinement& refinement) const {
    std::unique_ptr<IResettableStatisticsSubset> statisticsSubsetPtr =
      createStatisticsSubset(statistics, outputIndices, missingFeatureVector);
    searchForNominalRefinementInternally(featureVector, *statisticsSubsetPtr, comparator, numExamplesWithNonZeroWeights,
                                         minCoverage, refinement);
}

void FeatureBasedSearch::searchForBinaryRefinement(
  const BinaryFeatureVector& featureVector, const MissingFeatureVector& missingFeatureVector,
  SingleRefinementComparator& comparator, const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
  uint32 numExamplesWithNonZeroWeights, uint32 minCoverage, Refinement& refinement) const {
    std::unique_ptr<IResettableStatisticsSubset> statisticsSubsetPtr =
      createStatisticsSubset(statistics, outputIndices, missingFeatureVector);
    searchForBinaryRefinementInternally(featureVector, *statisticsSubsetPtr, comparator, numExamplesWithNonZeroWeights,
                                        minCoverage, refinement);
}

void FeatureBasedSearch::searchForBinaryRefinement(
  const BinaryFeatureVector& featureVector, const MissingFeatureVector& missingFeatureVector,
  FixedRefinementComparator& comparator, const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
  uint32 numExamplesWithNonZeroWeights, uint32 minCoverage, Refinement& refinement) const {
    std::unique_ptr<IResettableStatisticsSubset> statisticsSubsetPtr =
      createStatisticsSubset(statistics, outputIndices, missingFeatureVector);
    searchForBinaryRefinementInternally(featureVector, *statisticsSubsetPtr, comparator, numExamplesWithNonZeroWeights,
                                        minCoverage, refinement);
}

void FeatureBasedSearch::searchForOrdinalRefinement(
  const OrdinalFeatureVector& featureVector, const MissingFeatureVector& missingFeatureVector,
  SingleRefinementComparator& comparator, const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
  uint32 numExamplesWithNonZeroWeights, uint32 minCoverage, Refinement& refinement) const {
    std::unique_ptr<IResettableStatisticsSubset> statisticsSubsetPtr =
      createStatisticsSubset(statistics, outputIndices, missingFeatureVector);
    searchForOrdinalRefinementInternally(featureVector, *statisticsSubsetPtr, comparator, numExamplesWithNonZeroWeights,
                                         minCoverage, refinement);
}

void FeatureBasedSearch::searchForOrdinalRefinement(
  const OrdinalFeatureVector& featureVector, const MissingFeatureVector& missingFeatureVector,
  FixedRefinementComparator& comparator, const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
  uint32 numExamplesWithNonZeroWeights, uint32 minCoverage, Refinement& refinement) const {
    std::unique_ptr<IResettableStatisticsSubset> statisticsSubsetPtr =
      createStatisticsSubset(statistics, outputIndices, missingFeatureVector);
    searchForOrdinalRefinementInternally(featureVector, *statisticsSubsetPtr, comparator, numExamplesWithNonZeroWeights,
                                         minCoverage, refinement);
}

void FeatureBasedSearch::searchForBinnedRefinement(
  const BinnedFeatureVector& featureVector, const MissingFeatureVector& missingFeatureVector,
  SingleRefinementComparator& comparator, const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
  uint32 numExamplesWithNonZeroWeights, uint32 minCoverage, Refinement& refinement) const {
    std::unique_ptr<IResettableStatisticsSubset> statisticsSubsetPtr =
      createStatisticsSubset(statistics, outputIndices, missingFeatureVector);
    searchForBinnedRefinementInternally(featureVector, *statisticsSubsetPtr, comparator, numExamplesWithNonZeroWeights,
                                        minCoverage, refinement);
}

void FeatureBasedSearch::searchForBinnedRefinement(
  const BinnedFeatureVector& featureVector, const MissingFeatureVector& missingFeatureVector,
  FixedRefinementComparator& comparator, const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
  uint32 numExamplesWithNonZeroWeights, uint32 minCoverage, Refinement& refinement) const {
    std::unique_ptr<IResettableStatisticsSubset> statisticsSubsetPtr =
      createStatisticsSubset(statistics, outputIndices, missingFeatureVector);
    searchForBinnedRefinementInternally(featureVector, *statisticsSubsetPtr, comparator, numExamplesWithNonZeroWeights,
                                        minCoverage, refinement);
}
