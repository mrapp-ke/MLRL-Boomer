#include "mlrl/common/rule_refinement/feature_based_search.hpp"

#include "feature_based_search_binary.hpp"
#include "feature_based_search_binned.hpp"
#include "feature_based_search_nominal.hpp"
#include "feature_based_search_numerical.hpp"
#include "feature_based_search_ordinal.hpp"

static inline void addMissingStatistics(IWeightedStatisticsSubset& statisticsSubset,
                                        const MissingFeatureVector& missingFeatureVector) {
    for (auto it = missingFeatureVector.indices_cbegin(); it != missingFeatureVector.indices_cend(); it++) {
        uint32 index = *it;
        statisticsSubset.addToMissing(index);
    }
}

void FeatureBasedSearch::searchForNumericalRefinement(const NumericalFeatureVector& featureVector,
                                                      const MissingFeatureVector& missingFeatureVector,
                                                      IWeightedStatisticsSubset& statisticsSubset,
                                                      SingleRefinementComparator& comparator,
                                                      uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                                      Refinement& refinement) const {
    addMissingStatistics(statisticsSubset, missingFeatureVector);
    searchForNumericalRefinementInternally(featureVector, statisticsSubset, comparator, numExamplesWithNonZeroWeights,
                                           minCoverage, refinement);
}

void FeatureBasedSearch::searchForNumericalRefinement(const NumericalFeatureVector& featureVector,
                                                      const MissingFeatureVector& missingFeatureVector,
                                                      IWeightedStatisticsSubset& statisticsSubset,
                                                      FixedRefinementComparator& comparator,
                                                      uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                                      Refinement& refinement) const {
    addMissingStatistics(statisticsSubset, missingFeatureVector);
    searchForNumericalRefinementInternally(featureVector, statisticsSubset, comparator, numExamplesWithNonZeroWeights,
                                           minCoverage, refinement);
}

void FeatureBasedSearch::searchForNominalRefinement(const NominalFeatureVector& featureVector,
                                                    const MissingFeatureVector& missingFeatureVector,
                                                    IWeightedStatisticsSubset& statisticsSubset,
                                                    SingleRefinementComparator& comparator,
                                                    uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                                    Refinement& refinement) const {
    addMissingStatistics(statisticsSubset, missingFeatureVector);
    searchForNominalRefinementInternally(featureVector, statisticsSubset, comparator, numExamplesWithNonZeroWeights,
                                         minCoverage, refinement);
}

void FeatureBasedSearch::searchForNominalRefinement(const NominalFeatureVector& featureVector,
                                                    const MissingFeatureVector& missingFeatureVector,
                                                    IWeightedStatisticsSubset& statisticsSubset,
                                                    FixedRefinementComparator& comparator,
                                                    uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                                    Refinement& refinement) const {
    addMissingStatistics(statisticsSubset, missingFeatureVector);
    searchForNominalRefinementInternally(featureVector, statisticsSubset, comparator, numExamplesWithNonZeroWeights,
                                         minCoverage, refinement);
}

void FeatureBasedSearch::searchForBinaryRefinement(const BinaryFeatureVector& featureVector,
                                                   const MissingFeatureVector& missingFeatureVector,
                                                   IWeightedStatisticsSubset& statisticsSubset,
                                                   SingleRefinementComparator& comparator,
                                                   uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                                   Refinement& refinement) const {
    addMissingStatistics(statisticsSubset, missingFeatureVector);
    searchForBinaryRefinementInternally(featureVector, statisticsSubset, comparator, numExamplesWithNonZeroWeights,
                                        minCoverage, refinement);
}

void FeatureBasedSearch::searchForBinaryRefinement(const BinaryFeatureVector& featureVector,
                                                   const MissingFeatureVector& missingFeatureVector,
                                                   IWeightedStatisticsSubset& statisticsSubset,
                                                   FixedRefinementComparator& comparator,
                                                   uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                                   Refinement& refinement) const {
    addMissingStatistics(statisticsSubset, missingFeatureVector);
    searchForBinaryRefinementInternally(featureVector, statisticsSubset, comparator, numExamplesWithNonZeroWeights,
                                        minCoverage, refinement);
}

void FeatureBasedSearch::searchForOrdinalRefinement(const OrdinalFeatureVector& featureVector,
                                                    const MissingFeatureVector& missingFeatureVector,
                                                    IWeightedStatisticsSubset& statisticsSubset,
                                                    SingleRefinementComparator& comparator,
                                                    uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                                    Refinement& refinement) const {
    addMissingStatistics(statisticsSubset, missingFeatureVector);
    searchForOrdinalRefinementInternally(featureVector, statisticsSubset, comparator, numExamplesWithNonZeroWeights,
                                         minCoverage, refinement);
}

void FeatureBasedSearch::searchForOrdinalRefinement(const OrdinalFeatureVector& featureVector,
                                                    const MissingFeatureVector& missingFeatureVector,
                                                    IWeightedStatisticsSubset& statisticsSubset,
                                                    FixedRefinementComparator& comparator,
                                                    uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                                    Refinement& refinement) const {
    addMissingStatistics(statisticsSubset, missingFeatureVector);
    searchForOrdinalRefinementInternally(featureVector, statisticsSubset, comparator, numExamplesWithNonZeroWeights,
                                         minCoverage, refinement);
}

void FeatureBasedSearch::searchForBinnedRefinement(const BinnedFeatureVector& featureVector,
                                                   const MissingFeatureVector& missingFeatureVector,
                                                   IWeightedStatisticsSubset& statisticsSubset,
                                                   SingleRefinementComparator& comparator,
                                                   uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                                   Refinement& refinement) const {
    addMissingStatistics(statisticsSubset, missingFeatureVector);
    searchForBinnedRefinementInternally(featureVector, statisticsSubset, comparator, numExamplesWithNonZeroWeights,
                                        minCoverage, refinement);
}

void FeatureBasedSearch::searchForBinnedRefinement(const BinnedFeatureVector& featureVector,
                                                   const MissingFeatureVector& missingFeatureVector,
                                                   IWeightedStatisticsSubset& statisticsSubset,
                                                   FixedRefinementComparator& comparator,
                                                   uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                                   Refinement& refinement) const {
    addMissingStatistics(statisticsSubset, missingFeatureVector);
    searchForBinnedRefinementInternally(featureVector, statisticsSubset, comparator, numExamplesWithNonZeroWeights,
                                        minCoverage, refinement);
}
