/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"
#include "mlrl/common/statistics/statistics_subset_weighted.hpp"

#include <memory>

/**
 * Defines an interface for all classes that provide access to weighted statistics about the quality of predictions for
 * training examples, which serve as the basis for learning a new rule or refining an existing one.
 */
class IImmutableWeightedStatistics {
    public:

        virtual ~IImmutableWeightedStatistics() {}

        /**
         * Returns the number of available statistics.
         *
         * @return The number of statistics
         */
        virtual uint32 getNumStatistics() const = 0;

        /**
         * Returns the number of available outputs.
         *
         * @return The number of outputs
         */
        virtual uint32 getNumOutputs() const = 0;

        /**
         * Creates and returns a new object of type `IWeightedStatisticsSubset` that includes only those outputs, whose
         * indices are provided by a specific `CompleteIndexVector`.
         *
         * @param outputIndices A reference to an object of type `CompleteIndexVector` that provides access to the
         *                      indices of the outputs that should be included in the subset
         * @return              An unique pointer to an object of type `IWeightedStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IWeightedStatisticsSubset> createSubset(
          const CompleteIndexVector& outputIndices) const = 0;

        /**
         * Creates and returns a new object of type `IWeightedStatisticsSubset` that includes only those outputs, whose
         * indices are provided by a specific `PartialIndexVector`.
         *
         * @param outputIndices A reference to an object of type `PartialIndexVector` that provides access to the
         *                      indices of the outputs that should be included in the subset
         * @return              An unique pointer to an object of type `IWeightedStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IWeightedStatisticsSubset> createSubset(
          const PartialIndexVector& outputIndices) const = 0;
};
