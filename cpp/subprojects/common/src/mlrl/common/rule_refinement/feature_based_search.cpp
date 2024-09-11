#include "mlrl/common/rule_refinement/feature_based_search.hpp"

#include "feature_based_search_binary.hpp"
#include "feature_based_search_binned.hpp"
#include "feature_based_search_nominal.hpp"
#include "feature_based_search_numerical.hpp"
#include "feature_based_search_ordinal.hpp"

static inline std::unique_ptr<IWeightedStatisticsSubset> createStatisticsSubset(const IWeightedStatistics& statistics,
                                                                                const IIndexVector& outputIndices) {
    std::unique_ptr<IWeightedStatisticsSubset> statisticsSubsetPtr;
    auto partialIndexVectorVisitor = [&](const PartialIndexVector& partialIndexVector) {
        statisticsSubsetPtr = statistics.createSubset(partialIndexVector);
    };
    auto completeIndexVectorVisitor = [&](const CompleteIndexVector& completeIndexVector) {
        statisticsSubsetPtr = statistics.createSubset(completeIndexVector);
    };
    outputIndices.visit(partialIndexVectorVisitor, completeIndexVectorVisitor);
    return statisticsSubsetPtr;
}

static inline void addMissingStatistics(IWeightedStatisticsSubset& statisticsSubset,
                                        const MissingFeatureVector& missingFeatureVector) {
    for (auto it = missingFeatureVector.indices_cbegin(); it != missingFeatureVector.indices_cend(); it++) {
        uint32 index = *it;
        statisticsSubset.addToMissing(index);
    }
}

void FeatureBasedSearch::searchForNumericalRefinement(
  const NumericalFeatureVector& featureVector, const MissingFeatureVector& missingFeatureVector,
  SingleRefinementComparator& comparator, const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
  uint32 numExamplesWithNonZeroWeights, uint32 minCoverage, Refinement& refinement) const {
    std::unique_ptr<IWeightedStatisticsSubset> statisticsSubsetPtr = createStatisticsSubset(statistics, outputIndices);
    addMissingStatistics(*statisticsSubsetPtr, missingFeatureVector);
    searchForNumericalRefinementInternally(featureVector, *statisticsSubsetPtr, comparator,
                                           numExamplesWithNonZeroWeights, minCoverage, refinement);
}

void FeatureBasedSearch::searchForNumericalRefinement(
  const NumericalFeatureVector& featureVector, const MissingFeatureVector& missingFeatureVector,
  FixedRefinementComparator& comparator, const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
  uint32 numExamplesWithNonZeroWeights, uint32 minCoverage, Refinement& refinement) const {
    std::unique_ptr<IWeightedStatisticsSubset> statisticsSubsetPtr = createStatisticsSubset(statistics, outputIndices);
    addMissingStatistics(*statisticsSubsetPtr, missingFeatureVector);
    searchForNumericalRefinementInternally(featureVector, *statisticsSubsetPtr, comparator,
                                           numExamplesWithNonZeroWeights, minCoverage, refinement);
}

void FeatureBasedSearch::searchForNominalRefinement(
  const NominalFeatureVector& featureVector, const MissingFeatureVector& missingFeatureVector,
  SingleRefinementComparator& comparator, const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
  uint32 numExamplesWithNonZeroWeights, uint32 minCoverage, Refinement& refinement) const {
    std::unique_ptr<IWeightedStatisticsSubset> statisticsSubsetPtr = createStatisticsSubset(statistics, outputIndices);
    addMissingStatistics(*statisticsSubsetPtr, missingFeatureVector);
    searchForNominalRefinementInternally(featureVector, *statisticsSubsetPtr, comparator, numExamplesWithNonZeroWeights,
                                         minCoverage, refinement);
}

void FeatureBasedSearch::searchForNominalRefinement(
  const NominalFeatureVector& featureVector, const MissingFeatureVector& missingFeatureVector,
  FixedRefinementComparator& comparator, const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
  uint32 numExamplesWithNonZeroWeights, uint32 minCoverage, Refinement& refinement) const {
    std::unique_ptr<IWeightedStatisticsSubset> statisticsSubsetPtr = createStatisticsSubset(statistics, outputIndices);
    addMissingStatistics(*statisticsSubsetPtr, missingFeatureVector);
    searchForNominalRefinementInternally(featureVector, *statisticsSubsetPtr, comparator, numExamplesWithNonZeroWeights,
                                         minCoverage, refinement);
}

void FeatureBasedSearch::searchForBinaryRefinement(
  const BinaryFeatureVector& featureVector, const MissingFeatureVector& missingFeatureVector,
  SingleRefinementComparator& comparator, const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
  uint32 numExamplesWithNonZeroWeights, uint32 minCoverage, Refinement& refinement) const {
    std::unique_ptr<IWeightedStatisticsSubset> statisticsSubsetPtr = createStatisticsSubset(statistics, outputIndices);
    addMissingStatistics(*statisticsSubsetPtr, missingFeatureVector);
    searchForBinaryRefinementInternally(featureVector, *statisticsSubsetPtr, comparator, numExamplesWithNonZeroWeights,
                                        minCoverage, refinement);
}

void FeatureBasedSearch::searchForBinaryRefinement(
  const BinaryFeatureVector& featureVector, const MissingFeatureVector& missingFeatureVector,
  FixedRefinementComparator& comparator, const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
  uint32 numExamplesWithNonZeroWeights, uint32 minCoverage, Refinement& refinement) const {
    std::unique_ptr<IWeightedStatisticsSubset> statisticsSubsetPtr = createStatisticsSubset(statistics, outputIndices);
    addMissingStatistics(*statisticsSubsetPtr, missingFeatureVector);
    searchForBinaryRefinementInternally(featureVector, *statisticsSubsetPtr, comparator, numExamplesWithNonZeroWeights,
                                        minCoverage, refinement);
}

void FeatureBasedSearch::searchForOrdinalRefinement(
  const OrdinalFeatureVector& featureVector, const MissingFeatureVector& missingFeatureVector,
  SingleRefinementComparator& comparator, const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
  uint32 numExamplesWithNonZeroWeights, uint32 minCoverage, Refinement& refinement) const {
    std::unique_ptr<IWeightedStatisticsSubset> statisticsSubsetPtr = createStatisticsSubset(statistics, outputIndices);
    addMissingStatistics(*statisticsSubsetPtr, missingFeatureVector);
    searchForOrdinalRefinementInternally(featureVector, *statisticsSubsetPtr, comparator, numExamplesWithNonZeroWeights,
                                         minCoverage, refinement);
}

void FeatureBasedSearch::searchForOrdinalRefinement(
  const OrdinalFeatureVector& featureVector, const MissingFeatureVector& missingFeatureVector,
  FixedRefinementComparator& comparator, const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
  uint32 numExamplesWithNonZeroWeights, uint32 minCoverage, Refinement& refinement) const {
    std::unique_ptr<IWeightedStatisticsSubset> statisticsSubsetPtr = createStatisticsSubset(statistics, outputIndices);
    addMissingStatistics(*statisticsSubsetPtr, missingFeatureVector);
    searchForOrdinalRefinementInternally(featureVector, *statisticsSubsetPtr, comparator, numExamplesWithNonZeroWeights,
                                         minCoverage, refinement);
}

void FeatureBasedSearch::searchForBinnedRefinement(
  const BinnedFeatureVector& featureVector, const MissingFeatureVector& missingFeatureVector,
  SingleRefinementComparator& comparator, const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
  uint32 numExamplesWithNonZeroWeights, uint32 minCoverage, Refinement& refinement) const {
    std::unique_ptr<IWeightedStatisticsSubset> statisticsSubsetPtr = createStatisticsSubset(statistics, outputIndices);
    addMissingStatistics(*statisticsSubsetPtr, missingFeatureVector);
    searchForBinnedRefinementInternally(featureVector, *statisticsSubsetPtr, comparator, numExamplesWithNonZeroWeights,
                                        minCoverage, refinement);
}

void FeatureBasedSearch::searchForBinnedRefinement(
  const BinnedFeatureVector& featureVector, const MissingFeatureVector& missingFeatureVector,
  FixedRefinementComparator& comparator, const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
  uint32 numExamplesWithNonZeroWeights, uint32 minCoverage, Refinement& refinement) const {
    std::unique_ptr<IWeightedStatisticsSubset> statisticsSubsetPtr = createStatisticsSubset(statistics, outputIndices);
    addMissingStatistics(*statisticsSubsetPtr, missingFeatureVector);
    searchForBinnedRefinementInternally(featureVector, *statisticsSubsetPtr, comparator, numExamplesWithNonZeroWeights,
                                        minCoverage, refinement);
}
