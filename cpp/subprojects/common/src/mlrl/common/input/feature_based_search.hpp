/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_based_search_binary.hpp"
#include "feature_based_search_binned.hpp"
#include "feature_based_search_nominal.hpp"
#include "feature_based_search_numerical.hpp"
#include "feature_based_search_ordinal.hpp"
#include "mlrl/common/input/feature_vector_binary.hpp"
#include "mlrl/common/input/feature_vector_binned.hpp"
#include "mlrl/common/input/feature_vector_missing.hpp"
#include "mlrl/common/input/feature_vector_nominal.hpp"
#include "mlrl/common/input/feature_vector_numerical.hpp"
#include "mlrl/common/input/feature_vector_ordinal.hpp"
#include "mlrl/common/rule_refinement/refinement_comparator_fixed.hpp"
#include "mlrl/common/rule_refinement/refinement_comparator_single.hpp"
#include "mlrl/common/statistics/statistics_weighted.hpp"

#include <memory>

/**
 * Creates and returns a new object of type `IResettableStatisticsSubset`, based on given `IWeightedStatistics`.
 *
 * @param statistics                A reference to an object of type `IWeightedStatistics`
 * @param excludedStatisticIndices  A reference to an object of type `BinaryDokVector` that provides access to the
 *                                  indices of the statistics that should be excluded from the subset
 * @param outputIndices             A reference to an object of type `IIndexVector` that provides access to the indices
 *                                  of the outputs that should be included in the subset
 * @return                          An unique pointer to an object of type `IResettableStatisticsSubset` that has been
 *                                  created
 */
static inline std::unique_ptr<IResettableStatisticsSubset> createStatisticsSubset(
  const IWeightedStatistics& statistics, const BinaryDokVector& excludedStatisticIndices,
  const IIndexVector& outputIndices) {
    std::unique_ptr<IResettableStatisticsSubset> statisticsSubsetPtr;
    auto partialIndexVectorVisitor = [&](const PartialIndexVector& partialIndexVector) {
        statisticsSubsetPtr = statistics.createSubset(excludedStatisticIndices, partialIndexVector);
    };
    auto completeIndexVectorVisitor = [&](const CompleteIndexVector& completeIndexVector) {
        statisticsSubsetPtr = statistics.createSubset(excludedStatisticIndices, completeIndexVector);
    };
    outputIndices.visit(partialIndexVectorVisitor, completeIndexVectorVisitor);
    return statisticsSubsetPtr;
}

/**
 * Conducts a search for the best refinement of an existing rule that can be created from a `NumericalFeatureVector`.
 *
 * @tparam Comparator                   The type of the comparator that should be used for comparing potential
 *                                      refinements
 * @param featureVector                 A reference to an object of type `NumericalFeatureVector`, the refinements
 *                                      should be created from
 * @param missingFeatureVector          A reference to an object of type `MissingFeatureVector` that provides access to
 *                                      the indices of training examples with missing feature values
 * @param comparator                    A reference to an object of template type `Comparator` that should be used for
 *                                      comparing potential refinements
 * @param statistics                    A reference to an object of type `IWeightedStatistics` that provides access to
 *                                      weighted statistics about the quality of predictions for training examples,
 *                                      which should serve as the basis for evaluating the quality of potential
 *                                      refinements
 * @param outputIndices                 A reference to an object of type `IIndexVector` that provides access to the
 *                                      indices of the outputs for which refinements should predict
 * @param numExamplesWithNonZeroWeights The total number of examples with non-zero weights that may be covered by a
 *                                      refinement
 * @param minCoverage                   The minimum number of examples that must be covered by the refinement
 * @param refinement                    A reference to an object of type `Refinement` that should be used for
 *                                      storing the properties of the best refinement that is found
 */
template<typename Comparator>
static inline void searchForNumericalRefinement(const NumericalFeatureVector& featureVector,
                                                const MissingFeatureVector& missingFeatureVector,
                                                Comparator& comparator, const IWeightedStatistics& statistics,
                                                const IIndexVector& outputIndices, uint32 numExamplesWithNonZeroWeights,
                                                uint32 minCoverage, Refinement& refinement) {
    std::unique_ptr<IResettableStatisticsSubset> statisticsSubsetPtr =
      createStatisticsSubset(statistics, missingFeatureVector, outputIndices);
    searchForNumericalRefinementInternally(featureVector, *statisticsSubsetPtr, comparator,
                                           numExamplesWithNonZeroWeights, minCoverage, refinement);
}

/**
 * Conducts a search for the best refinement of an existing rule that can be created from a `NominalFeatureVector`.
 *
 * @tparam Comparator                   The type of the comparator that should be used for comparing potential
 *                                      refinements
 * @param featureVector                 A reference to an object of type `NominalFeatureVector`, the refinements should
 *                                      be created from
 * @param missingFeatureVector          A reference to an object of type `MissingFeatureVector` that provides access to
 *                                      the indices of training examples with missing feature values
 * @param comparator                    A reference to an object of template type `Comparator` that should be used for
 *                                      comparing potential refinements
 * @param statistics                    A reference to an object of type `IWeightedStatistics` that provides access to
 *                                      weighted statistics about the quality of predictions for training examples,
 *                                      which should serve as the basis for evaluating the quality of potential
 *                                      refinements
 * @param outputIndices                 A reference to an object of type `IIndexVector` that provides access to the
 *                                      indices of the outputs for which refinements should predict
 * @param numExamplesWithNonZeroWeights The total number of examples with non-zero weights that may be covered by a
 *                                      refinement
 * @param minCoverage                   The minimum number of examples that must be covered by the refinement
 * @param refinement                    A reference to an object of type `Refinement` that should be used for
 *                                      storing the properties of the best refinement that is found
 */
template<typename Comparator>
static inline void searchForNominalRefinement(const NominalFeatureVector& featureVector,
                                              const MissingFeatureVector& missingFeatureVector, Comparator& comparator,
                                              const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
                                              uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                              Refinement& refinement) {
    std::unique_ptr<IResettableStatisticsSubset> statisticsSubsetPtr =
      createStatisticsSubset(statistics, missingFeatureVector, outputIndices);
    searchForNominalRefinementInternally(featureVector, *statisticsSubsetPtr, comparator, numExamplesWithNonZeroWeights,
                                         minCoverage, refinement);
}

/**
 * Conducts a search for the best refinement of an existing rule that can be created from a `BinaryFeatureVector`.
 *
 *
 * @param featureVector                 A reference to an object of type `BinaryFeatureVector`, the refinements
 *                                      should be created from
 * @param missingFeatureVector          A reference to an object of type `MissingFeatureVector` that provides access to
 *                                      the indices of training examples with missing feature values
 * @param comparator                    A reference to an object of template type `Comparator` that should be used for
 *                                      comparing potential refinements
 * @param statistics                    A reference to an object of type `IWeightedStatistics` that provides access to
 *                                      weighted statistics about the quality of predictions for training examples,
 *                                      which should serve as the basis for evaluating the quality of potential
 *                                      refinements
 * @param outputIndices                 A reference to an object of type `IIndexVector` that provides access to the
 *                                      indices of the outputs for which refinements should predict
 * @param numExamplesWithNonZeroWeights The total number of examples with non-zero weights that may be covered by a
 *                                      refinement
 * @param minCoverage                   The minimum number of examples that must be covered by the refinement
 * @param refinement                    A reference to an object of type `Refinement` that should be used for
 *                                      storing the properties of the best refinement that is found
 */
template<typename Comparator>
static inline void searchForBinaryRefinement(const BinaryFeatureVector& featureVector,
                                             const MissingFeatureVector& missingFeatureVector, Comparator& comparator,
                                             const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
                                             uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                             Refinement& refinement) {
    std::unique_ptr<IResettableStatisticsSubset> statisticsSubsetPtr =
      createStatisticsSubset(statistics, missingFeatureVector, outputIndices);
    searchForBinaryRefinementInternally(featureVector, *statisticsSubsetPtr, comparator, numExamplesWithNonZeroWeights,
                                        minCoverage, refinement);
}

/**
 * Conducts a search for the best refinement of an existing rule that can be created from an `OrdinalFeatureVector`.
 *
 * @tparam Comparator                   The type of the comparator that should be used for comparing potential
 *                                      refinements
 * @param featureVector                 A reference to an object of type `OrdinalFeatureVector`, the refinements should
 *                                      be created from
 * @param missingFeatureVector          A reference to an object of type `MissingFeatureVector` that provides access to
 *                                      the indices of training examples with missing feature values
 * @param comparator                    A reference to an object of template type `Comparator` that should be used for
 *                                      comparing potential refinements
 * @param statistics                    A reference to an object of type `IWeightedStatistics` that provides access to
 *                                      weighted statistics about the quality of predictions for training examples,
 *                                      which should serve as the basis for evaluating the quality of potential
 *                                      refinements
 * @param outputIndices                 A reference to an object of type `IIndexVector` that provides access to the
 *                                      indices of the outputs for which refinements should predict
 * @param numExamplesWithNonZeroWeights The total number of examples with non-zero weights that may be covered by a
 *                                      refinement
 * @param minCoverage                   The minimum number of examples that must be covered by the refinement
 * @param refinement                    A reference to an object of type `Refinement` that should be used for
 *                                      storing the properties of the best refinement that is found
 */
template<typename Comparator>
static inline void searchForOrdinalRefinement(const OrdinalFeatureVector& featureVector,
                                              const MissingFeatureVector& missingFeatureVector, Comparator& comparator,
                                              const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
                                              uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                              Refinement& refinement) {
    std::unique_ptr<IResettableStatisticsSubset> statisticsSubsetPtr =
      createStatisticsSubset(statistics, missingFeatureVector, outputIndices);
    searchForOrdinalRefinementInternally(featureVector, *statisticsSubsetPtr, comparator, numExamplesWithNonZeroWeights,
                                         minCoverage, refinement);
}

/**
 * Conducts a search for the best refinement of an existing rule that can be created from a `BinnedFeatureVector`.
 *
 * @tparam Comparator                   The type of the comparator that should be used for comparing potential
 *                                      refinements
 * @param featureVector                 A reference to an object of type `BinnedFeatureVector`, the refinements should
 *                                      be created from
 * @param missingFeatureVector          A reference to an object of type `MissingFeatureVector` that provides access to
 *                                      the indices of training examples with missing feature values
 * @param comparator                    A reference to an object of template type `Comparator` that should be used for
 *                                      comparing potential refinements
 * @param statistics                    A reference to an object of type `IWeightedStatistics` that provides access to
 *                                      weighted statistics about the quality of predictions for training examples,
 *                                      which should serve as the basis for evaluating the quality of potential
 *                                      refinements
 * @param outputIndices                 A reference to an object of type `IIndexVector` that provides access to the
 *                                      indices of the outputs for which refinements should predict
 * @param numExamplesWithNonZeroWeights The total number of examples with non-zero weights that may be covered by a
 *                                      refinement
 * @param minCoverage                   The minimum number of examples that must be covered by the refinement
 * @param refinement                    A reference to an object of type `Refinement` that should be used for
 *                                      storing the properties of the best refinement that is found
 */
template<typename Comparator>
static inline void searchForBinnedRefinement(const BinnedFeatureVector& featureVector,
                                             const MissingFeatureVector& missingFeatureVector, Comparator& comparator,
                                             const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
                                             uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                             Refinement& refinement) {
    std::unique_ptr<IResettableStatisticsSubset> statisticsSubsetPtr =
      createStatisticsSubset(statistics, missingFeatureVector, outputIndices);
    searchForBinnedRefinementInternally(featureVector, *statisticsSubsetPtr, comparator, numExamplesWithNonZeroWeights,
                                        minCoverage, refinement);
}
