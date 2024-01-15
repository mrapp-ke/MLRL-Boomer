#include "mlrl/common/rule_refinement/rule_refinement_search.hpp"

#include "rule_refinement_search_binary.hpp"

static inline void addMissingStatistics(IWeightedStatisticsSubset& statisticsSubset,
                                        const MissingFeatureVector& missingFeatureVector) {
    for (auto it = missingFeatureVector.indices_cbegin(); it != missingFeatureVector.indices_cend(); it++) {
        uint32 index = *it;
        statisticsSubset.addToMissing(index);
    }
}

void RuleRefinementSearch::searchForNumericalRefinement(const NumericalFeatureVector& featureVector,
                                                        const MissingFeatureVector& missingFeatureVector,
                                                        IWeightedStatisticsSubset& statisticsSubset,
                                                        SingleRefinementComparator& comparator, uint32 minCoverage,
                                                        Refinement& refinement) const {
    addMissingStatistics(statisticsSubset, missingFeatureVector);
    // TODO Implement
}

void RuleRefinementSearch::searchForNumericalRefinement(const NumericalFeatureVector& featureVector,
                                                        const MissingFeatureVector& missingFeatureVector,
                                                        IWeightedStatisticsSubset& statisticsSubset,
                                                        FixedRefinementComparator& comparator, uint32 minCoverage,
                                                        Refinement& refinement) const {
    addMissingStatistics(statisticsSubset, missingFeatureVector);
    // TODO Implement
}

void RuleRefinementSearch::searchForNominalRefinement(const NominalFeatureVector& featureVector,
                                                      const MissingFeatureVector& missingFeatureVector,
                                                      IWeightedStatisticsSubset& statisticsSubset,
                                                      SingleRefinementComparator& comparator, uint32 minCoverage,
                                                      Refinement& refinement) const {
    addMissingStatistics(statisticsSubset, missingFeatureVector);
    // TODO Implement
}

void RuleRefinementSearch::searchForNominalRefinement(const NominalFeatureVector& featureVector,
                                                      const MissingFeatureVector& missingFeatureVector,
                                                      IWeightedStatisticsSubset& statisticsSubset,
                                                      FixedRefinementComparator& comparator, uint32 minCoverage,
                                                      Refinement& refinement) const {
    addMissingStatistics(statisticsSubset, missingFeatureVector);
    // TODO Implement
}

void RuleRefinementSearch::searchForBinaryRefinement(const BinaryFeatureVector& featureVector,
                                                     const MissingFeatureVector& missingFeatureVector,
                                                     IWeightedStatisticsSubset& statisticsSubset,
                                                     SingleRefinementComparator& comparator, uint32 minCoverage,
                                                     Refinement& refinement) const {
    addMissingStatistics(statisticsSubset, missingFeatureVector);
    searchForBinaryRefinementInternally(featureVector, statisticsSubset, comparator, minCoverage, refinement);
}

void RuleRefinementSearch::searchForBinaryRefinement(const BinaryFeatureVector& featureVector,
                                                     const MissingFeatureVector& missingFeatureVector,
                                                     IWeightedStatisticsSubset& statisticsSubset,
                                                     FixedRefinementComparator& comparator, uint32 minCoverage,
                                                     Refinement& refinement) const {
    addMissingStatistics(statisticsSubset, missingFeatureVector);
    searchForBinaryRefinementInternally(featureVector, statisticsSubset, comparator, minCoverage, refinement);
}

void RuleRefinementSearch::searchForOrdinalRefinement(const OrdinalFeatureVector& featureVector,
                                                      const MissingFeatureVector& missingFeatureVector,
                                                      IWeightedStatisticsSubset& statisticsSubset,
                                                      SingleRefinementComparator& comparator, uint32 minCoverage,
                                                      Refinement& refinement) const {
    addMissingStatistics(statisticsSubset, missingFeatureVector);
    // TODO Implement
}

void RuleRefinementSearch::searchForOrdinalRefinement(const OrdinalFeatureVector& featureVector,
                                                      const MissingFeatureVector& missingFeatureVector,
                                                      IWeightedStatisticsSubset& statisticsSubset,
                                                      FixedRefinementComparator& comparator, uint32 minCoverage,
                                                      Refinement& refinement) const {
    addMissingStatistics(statisticsSubset, missingFeatureVector);
    // TODO Implement
}
