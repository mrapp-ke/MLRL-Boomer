/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/interval.hpp"
#include "mlrl/common/rule_refinement/coverage_mask.hpp"
#include "mlrl/common/rule_refinement/refinement_comparator_fixed.hpp"
#include "mlrl/common/rule_refinement/refinement_comparator_single.hpp"
#include "mlrl/common/statistics/statistics_weighted.hpp"

#include <memory>

/**
 * Defines an interface for all one-dimensional vectors that store the values of training examples for a certain
 * feature.
 */
class IFeatureVector {
    public:

        virtual ~IFeatureVector() {}

        /**
         * Creates and returns an `IResettableStatisticsSubset` based on given statistics and this feature vector.
         *
         * @param statistics    A reference to an object of type `IWeightedStatisticss`, the subset should be based on
         * @param outputIndices A reference to an object of type `CompleteIndexVector` that provides access to the
         *                      indices of the outputs that should be included in the subset
         * @return              An unique pointer to an object of type `IResettableStatisticsSubset` that has been
         *                      created
         */
        virtual std::unique_ptr<IResettableStatisticsSubset> createStatisticsSubset(
          const IWeightedStatistics& statistics, const CompleteIndexVector& outputIndices) const = 0;

        /**
         * Creates and returns an `IResettableStatisticsSubset` based on given statistics and this feature vector.
         *
         * @param statistics    A reference to an object of type `IWeightedStatisticss`, the subset should be based on
         * @param outputIndices A reference to an object of type `PartialIndexVector` that provides access to the
         *                      indices of the outputs that should be included in the subset
         * @return              An unique pointer to an object of type `IResettableStatisticsSubset` that has been
         *                      created
         */
        virtual std::unique_ptr<IResettableStatisticsSubset> createStatisticsSubset(
          const IWeightedStatistics& statistics, const PartialIndexVector& outputIndices) const = 0;

        /**
         * Conducts a search for the best refinement of an existing rule that can be created from a this feature vector.
         *
         * @param comparator                    A reference to an object of type `SingleRefinementComparator` that
         *                                      should be used for comparing potential refinements
         * @param statistics                    A reference to an object of type `IWeightedStatistics` that provides
         *                                      access to weighted statistics about the quality of predictions for
         *                                      training examples, which should serve as the basis for evaluating the
         *                                      quality of potential refinements
         * @param outputIndices                 A reference to an object of type `IIndexVector` that provides access to
         *                                      the indices of the outputs for which refinements should predict
         * @param numExamplesWithNonZeroWeights The total number of examples with non-zero weights that may be covered
         *                                      by a refinement
         * @param minCoverage                   The minimum number of examples that must be covered by the refinement
         * @param refinement                    A reference to an object of type `Refinement` that should be used for
         *                                      storing the properties of the best refinement that is found
         */
        virtual void searchForRefinement(SingleRefinementComparator& comparator, const IWeightedStatistics& statistics,
                                         const IIndexVector& outputIndices, uint32 numExamplesWithNonZeroWeights,
                                         uint32 minCoverage, Refinement& refinement) const = 0;

        /**
         * Conducts a search for the best refinement of an existing rule that can be created from a this feature vector.
         *
         * @param comparator                    A reference to an object of type `MultiRefinementComparator` that should
         *                                      be used for comparing potential refinements
         * @param statistics                    A reference to an object of type `IWeightedStatistics` that provides
         *                                      access to weighted statistics about the quality of predictions for
         *                                      training examples, which should serve as the basis for evaluating the
         *                                      quality of potential refinements
         * @param outputIndices                 A reference to an object of type `IIndexVector` that provides access to
         *                                      the indices of the outputs for which refinements should predict
         * @param numExamplesWithNonZeroWeights The total number of examples with non-zero weights that may be covered
         *                                      by a refinement
         * @param minCoverage                   The minimum number of examples that must be covered by the refinement
         * @param refinement                    A reference to an object of type `Refinement` that should be used for
         *                                      storing the properties of the best refinement that is found
         */
        virtual void searchForRefinement(FixedRefinementComparator& comparator, const IWeightedStatistics& statistics,
                                         const IIndexVector& outputIndices, uint32 numExamplesWithNonZeroWeights,
                                         uint32 minCoverage, Refinement& refinement) const = 0;

        /**
         * Updates a given `CoverageMask` and `IWeightedStatistics` depending on the indices of training examples
         * included in a specific `Interval`.
         *
         * @param interval        A reference to an object of type `Interval` that includes the indices of all covered
         *                        training examples
         * @param coverageMask    A reference to an object of type `CoverageMask` that should be updated
         * @param indicatorValue  The indicator value that should be used for updating the given `CoverageMask`
         * @param statistics      A reference to an object of type `IWeightedStatistics` that should be updated
         */
        virtual void updateCoverageMaskAndStatistics(const Interval& interval, CoverageMask& coverageMask,
                                                     uint32 indicatorValue, IWeightedStatistics& statistics) const = 0;

        /**
         * Creates and returns a copy of this vector that does only store the feature values of training examples
         * included in a given `Interval`.
         *
         * @param existing  A reference to an unique pointer that stores an object of type `IFeatureVector` that may be
         *                  reused or a null pointer, if no such object is available
         * @param interval  A reference to an object of type `Interval` that includes the indices of the training
         *                  examples to be retained
         * @return          An unique pointer to an object of type `IFeatureVector` that has been created
         */
        virtual std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                            const Interval& interval) const = 0;

        /**
         * Creates and returns a copy of this vector that does only store the feature values of training examples marked
         * as covered according to a given `CoverageMask`.
         *
         * @param existing      A reference to an unique pointer that stores an object of type `IFeatureVector` that may
         *                      be reused or a null pointer, if no such object is available
         * @param coverageMask  A reference to an object of type `CoverageMask` that specifies the indices of the
         *                      training examples to be retained
         * @return              An unique pointer to an object of type `IFeatureVector` that has been created
         */
        virtual std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                            const CoverageMask& coverageMask) const = 0;
};
