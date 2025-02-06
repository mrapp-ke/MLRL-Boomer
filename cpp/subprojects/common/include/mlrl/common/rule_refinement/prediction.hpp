/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/indices/index_vector.hpp"
#include "mlrl/common/iterator/iterator_binned.hpp"
#include "mlrl/common/sampling/weight_vector_bit.hpp"
#include "mlrl/common/sampling/weight_vector_dense.hpp"
#include "mlrl/common/sampling/weight_vector_equal.hpp"
#include "mlrl/common/sampling/weight_vector_out_of_sample.hpp"

#include <memory>

// Forward declarations
class IPostProcessor;
class IStatistics;
class IStatisticsUpdate;
class IStatisticsSubset;
class IHead;

/**
 * Defines an interface for all classes that store the scores that are predicted by a rule.
 */
class IPrediction : public IIndexVector {
    public:

        virtual ~IPrediction() override {}

        /**
         * Sorts the scores that are stored by this prediction in increasing order by the indices of the outputs they
         * correspond to.
         */
        virtual void sort() = 0;

        /**
         * Post-processes the scores that are stored by this prediction.
         *
         * @param postProcessor A reference to an object of type `IPostProcessor` that should be used for
         *                      post-processing
         */
        virtual void postProcess(const IPostProcessor& postProcessor) = 0;

        /**
         * Creates and returns an object of type `IStatistics::IUpdate` that allows updating given `IStatistics` based
         * on this prediction.
         *
         * @param statistics  A reference to an object of type `IStatistics` that should be updated
         * @return            An unique pointer to an object of type `IStatistics::IUpdate` that has been created
         */
        virtual std::unique_ptr<IStatisticsUpdate> createStatisticsUpdate(IStatistics& statistics) const = 0;

        /**
         * Creates and returns a new subset of the given statistics that only contains the outputs whose indices are
         * stored in this vector.
         *
         * @param statistics    A reference to an object of type `IStatistics` that should be used to create the subset
         * @param weights       A reference to an object of type `EqualWeightVector` that provides access to the weights
         *                      of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createStatisticsSubset(const IStatistics& statistics,
                                                                          const EqualWeightVector& weights) const = 0;

        /**
         * Creates and returns a new subset of the given statistics that only contains the outputs whose indices are
         * stored in this vector.
         *
         * @param statistics    A reference to an object of type `IStatistics` that should be used to create the subset
         * @param weights       A reference to an object of type `BitWeightVector` that provides access to the weights
         *                      of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createStatisticsSubset(const IStatistics& statistics,
                                                                          const BitWeightVector& weights) const = 0;

        /**
         * Creates and returns a new subset of the given statistics that only contains the outputs whose indices are
         * stored in this vector.
         *
         * @param statistics    A reference to an object of type `IStatistics` that should be used to create the subset
         * @param weights       A reference to an object of type `DenseWeightVector<uint32>` that provides access to the
         *                      weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createStatisticsSubset(
          const IStatistics& statistics, const DenseWeightVector<uint32>& weights) const = 0;

        /**
         * Creates and returns a new subset of the given statistics that only contains the outputs whose indices are
         * stored in this vector.
         *
         * @param statistics    A reference to an object of type `IStatistics` that should be used to create the subset
         * @param weights       A reference to an object of type `DenseWeightVector<float32>` that provides access to
         *                      the weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createStatisticsSubset(
          const IStatistics& statistics, const DenseWeightVector<float32>& weights) const = 0;

        /**
         * Creates and returns a new subset of the given statistics that only contains the outputs whose indices are
         * stored in this vector.
         *
         * @param statistics    A reference to an object of type `IStatistics` that should be used to create the subset
         * @param weights       A reference to an object of type `OutOfSampleWeightVector<EqualWeightVector>` that
         *                      provides access to the weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createStatisticsSubset(
          const IStatistics& statistics, const OutOfSampleWeightVector<EqualWeightVector>& weights) const = 0;

        /**
         * Creates and returns a new subset of the given statistics that only contains the outputs whose indices are
         * stored in this vector.
         *
         * @param statistics    A reference to an object of type `IStatistics` that should be used to create the subset
         * @param weights       A reference to an object of type `OutOfSampleWeightVector<BitWeightVector>` that
         *                      provides access to the weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createStatisticsSubset(
          const IStatistics& statistics, const OutOfSampleWeightVector<BitWeightVector>& weights) const = 0;

        /**
         * Creates and returns a new subset of the given statistics that only contains the outputs whose indices are
         * stored in this vector.
         *
         * @param statistics    A reference to an object of type `IStatistics` that should be used to create the subset
         * @param weights       A reference to an object of type `OutOfSampleWeightVector<DenseWeightVector<uint32>>`
         *                      that provides access to the weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createStatisticsSubset(
          const IStatistics& statistics, const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const = 0;

        /**
         * Creates and returns a new subset of the given statistics that only contains the outputs whose indices are
         * stored in this vector.
         *
         * @param statistics    A reference to an object of type `IStatistics` that should be used to create the subset
         * @param weights       A reference to an object of type `OutOfSampleWeightVector<DenseWeightVector<float32>>`
         *                      that provides access to the weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createStatisticsSubset(
          const IStatistics& statistics, const OutOfSampleWeightVector<DenseWeightVector<float32>>& weights) const = 0;

        /**
         * Creates and returns a head that contains the scores that are stored by this prediction.
         *
         * @return An unique pointer to an object of type `IHead` that has been created
         */
        virtual std::unique_ptr<IHead> createHead() const = 0;
};
