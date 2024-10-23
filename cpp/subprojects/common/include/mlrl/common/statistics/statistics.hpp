/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 */
#pragma once

#include "mlrl/common/rule_refinement/prediction_complete.hpp"
#include "mlrl/common/rule_refinement/prediction_partial.hpp"
#include "mlrl/common/sampling/weight_vector_bit.hpp"
#include "mlrl/common/sampling/weight_vector_dense.hpp"
#include "mlrl/common/sampling/weight_vector_equal.hpp"
#include "mlrl/common/sampling/weight_vector_out_of_sample.hpp"
#include "mlrl/common/statistics/statistics_space.hpp"
#include "mlrl/common/statistics/statistics_weighted.hpp"

#include <memory>

/**
 * Defines an interface for all classes that allow updating statistics.
 */
class IStatisticsUpdate {
    public:

        virtual ~IStatisticsUpdate() {}

        /**
         * Updates a specific statistic.
         *
         * This function must be called for each statistic that is covered by a new rule before learning the
         * next rule.
         *
         * @param statisticIndex The index of the statistic that should be updated
         */
        virtual void applyPrediction(uint32 statisticIndex) = 0;

        /**
         * Reverts a specific statistic that has previously been updated via the function `applyPrediction`.
         *
         * @param statisticIndex The index of the statistic that should be updated
         */
        virtual void revertPrediction(uint32 statisticIndex) = 0;
};

/**
 * Defines an interface for all classes that provide access to statistics about the quality of predictions for training
 * examples, which serve as the basis for learning a new rule or refining an existing one.
 */
class IStatistics : public IStatisticsSpace {
    public:

        virtual ~IStatistics() override {}

        /**
         * Creates and returns a new object of type `IStatisticsUpdate` that allows updating the statistics based on the
         * predictions of a rule that predicts for all available outputs.
         *
         * @param prediction A reference to an object of type `CompletePrediction` that stores the scores that are
         *                   predicted by the rule
         */
        virtual std::unique_ptr<IStatisticsUpdate> createUpdate(const CompletePrediction& prediction) = 0;

        /**
         * Creates and returns a new object of type `IStatisticsUpdate` that allows updating the statistics based on the
         * predictions of a rule that predicts for a subset of the available outputs.
         *
         * @param prediction A reference to an object of type `PartialPrediction` that stores the scores that are
         *                   predicted by the rule
         */
        virtual std::unique_ptr<IStatisticsUpdate> createUpdate(const PartialPrediction& prediction) = 0;

        /**
         * Calculates and returns a numerical score that assesses the quality of the current predictions for a specific
         * statistic.
         *
         * @param statisticIndex    The index of the statistic for which the predictions should be evaluated
         * @return                  The numerical score that has been calculated
         */
        virtual float64 evaluatePrediction(uint32 statisticIndex) const = 0;

        /**
         * Creates and returns a new object of type `IStatisticsSubset` that includes only those outputs, whose indices
         * are provided by a specific `CompleteIndexVector`.
         *
         * @param outputIndices A reference to an object of type `CompleteIndexVector` that provides access to the
         *                      indices of the outputs that should be included in the subset
         * @param weights       A reference to an object of type `EqualWeightVector` that provides access to the weights
         *                      of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                                const EqualWeightVector& weights) const = 0;

        /**
         * Creates and returns a new object of type `IStatisticsSubset` that includes only those outputs, whose indices
         * are provided by a specific `PartialIndexVector`.
         *
         * @param outputIndices A reference to an object of type `PartialIndexVector` that provides access to the
         *                      indices of the outputs that should be included in the subset
         * @param weights       A reference to an object of type `EqualWeightVector` that provides access to the weights
         *                      of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                                const EqualWeightVector& weights) const = 0;

        /**
         * Creates and returns a new object of type `IStatisticsSubset` that includes only those outputs, whose indices
         * are provided by a specific `CompleteIndexVector`.
         *
         * @param outputIndices A reference to an object of type `CompleteIndexVector` that provides access to the
         *                      indices of the outputs that should be included in the subset
         * @param weights       A reference to an object of type `BitWeightVector` that provides access to the weights
         *                      of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                                const BitWeightVector& weights) const = 0;

        /**
         * Creates and returns a new object of type `IStatisticsSubset` that includes only those outputs, whose indices
         * are provided by a specific `PartialIndexVector`.
         *
         * @param outputIndices A reference to an object of type `PartialIndexVector` that provides access to the
         *                      indices of the outputs that should be included in the subset
         * @param weights       A reference to an object of type `BitWeightVector` that provides access to the weights
         *                      of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                                const BitWeightVector& weights) const = 0;

        /**
         * Creates and returns a new object of type `IStatisticsSubset` that includes only those outputs, whose indices
         * are provided by a specific `CompleteIndexVector`.
         *
         * @param outputIndices A reference to an object of type `CompleteIndexVector` that provides access to the
         *                      indices of the outputs that should be included in the subset
         * @param weights       A reference to an object of type `DenseWeightVector<uint32>` that provides access to the
         *                      weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                                const DenseWeightVector<uint32>& weights) const = 0;

        /**
         * Creates and returns a new object of type `IStatisticsSubset` that includes only those outputs, whose indices
         * are provided by a specific `PartialIndexVector`.
         *
         * @param outputIndices A reference to an object of type `PartialIndexVector` that provides access to the
         *                      indices of the outputs that should be included in the subset
         * @param weights       A reference to an object of type `DenseWeightVector<uint32>` that provides access to the
         *                      weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                                const DenseWeightVector<uint32>& weights) const = 0;

        /**
         * Creates and returns a new object of type `IStatisticsSubset` that includes only those outputs, whose indices
         * are provided by a specific `CompleteIndexVector`.
         *
         * @param outputIndices A reference to an object of type `CompleteIndexVector` that provides access to the
         *                      indices of the outputs that should be included in the subset
         * @param weights       A reference to an object of type `OutOfSampleWeightVector<EqualWeightVector>` that
         *                      provides access to the weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(
          const CompleteIndexVector& outputIndices,
          const OutOfSampleWeightVector<EqualWeightVector>& weights) const = 0;

        /**
         * Creates and returns a new object of type `IStatisticsSubset` that includes only those outputs, whose indices
         * are provided by a specific `PartialIndexVector`.
         *
         * @param outputIndices A reference to an object of type `PartialIndexVector` that provides access to the
         *                      indices of the outputs that should be included in the subset
         * @param weights       A reference to an object of type `OutOfSampleWeightVector<EqualWeightVector>` that
         *                      provides access to the weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(
          const PartialIndexVector& outputIndices, const OutOfSampleWeightVector<EqualWeightVector>& weights) const = 0;

        /**
         * Creates and returns a new object of type `IStatisticsSubset` that includes only those outputs, whose indices
         * are provided by a specific `CompleteIndexVector`.
         *
         * @param outputIndices A reference to an object of type `CompleteIndexVector` that provides access to the
         *                      indices of the outputs that should be included in the subset
         * @param weights       A reference to an object of type `OutOfSampleWeightVector<BitWeightVector>` that
         *                      provides access to the weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(
          const CompleteIndexVector& outputIndices, const OutOfSampleWeightVector<BitWeightVector>& weights) const = 0;

        /**
         * Creates and returns a new object of type `IStatisticsSubset` that includes only those outputs, whose indices
         * are provided by a specific `PartialIndexVector`.
         *
         * @param outputIndices A reference to an object of type `PartialIndexVector` that provides access to the
         *                      indices of the outputs that should be included in the subset
         * @param weights       A reference to an object of type `OutOfSampleWeightVector<BitWeightVector>` that
         *                      provides access to the weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(
          const PartialIndexVector& outputIndices, const OutOfSampleWeightVector<BitWeightVector>& weights) const = 0;

        /**
         * Creates and returns a new object of type `IStatisticsSubset` that includes only those outputs, whose indices
         * are provided by a specific `CompleteIndexVector`.
         *
         * @param outputIndices A reference to an object of type `CompleteIndexVector` that provides access to the
         *                      indices of the outputs that should be included in the subset
         * @param weights       A reference to an object of type `OutOfSampleWeightVector<DenseWeightVector<uint32>>`
         *                      that provides access to the weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(
          const CompleteIndexVector& outputIndices,
          const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const = 0;

        /**
         * Creates and returns a new object of type `IStatisticsSubset` that includes only those outputs, whose indices
         * are provided by a specific `PartialIndexVector`.
         *
         * @param outputIndices A reference to an object of type `PartialIndexVector` that provides access to the
         *                      indices of the outputs that should be included in the subset
         * @param weights       A reference to an object of type `OutOfSampleWeightVector<DenseWeightVector<uint32>>`
         *                      that provides access to the weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(
          const PartialIndexVector& outputIndices,
          const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const = 0;

        /**
         * Creates and returns a new object of type `IWeightedStatistics`.
         *
         * @param weights   A reference to an object of type `EqualWeightVector` that provides access to the weights of
         *                  individual training examples
         * @return          An unique pointer to an object of type `IWeightedStatistics` that has been created
         */
        virtual std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
          const EqualWeightVector& weights) const = 0;

        /**
         * Creates and returns a new object of type `IWeightedStatistics`.
         *
         * @param weights   A reference to an object of type `BitWeightVector` that provides access to the weights of
         *                  individual training examples
         * @return          An unique pointer to an object of type `IWeightedStatistics` that has been created
         */
        virtual std::unique_ptr<IWeightedStatistics> createWeightedStatistics(const BitWeightVector& weights) const = 0;

        /**
         * Creates and returns a new object of type `IWeightedStatistics`.
         *
         * @param weights   A reference to an object of type `DenseWeightVector<uint32>` that provides access to the
         *                  weights of individual training examples
         * @return          An unique pointer to an object of type `IWeightedStatistics` that has been created
         */
        virtual std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
          const DenseWeightVector<uint32>& weights) const = 0;
};
