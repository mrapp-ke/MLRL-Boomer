/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_dok_binary.hpp"
#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"
#include "mlrl/common/statistics/statistics_space.hpp"
#include "mlrl/common/statistics/statistics_subset_resettable.hpp"

#include <memory>

/**
 * Defines an interface for all classes that provide access to statistics about the quality of predictions for training
 * examples, which serve as the basis for learning a new rule or refining an existing one, and also provide functions
 * that allow to only use a sub-sample of the available statistics.
 */
class IWeightedStatistics : virtual public IStatisticsSpace {
    public:

        virtual ~IWeightedStatistics() override {}

        /**
         * Creates and returns a copy of this object.
         *
         * @return An unique pointer to an object of type `IWeightedStatistics` that has been created
         */
        virtual std::unique_ptr<IWeightedStatistics> copy() const = 0;

        /**
         * Resets the statistics which should be considered in the following for refining an existing rule. The indices
         * of the respective statistics must be provided via subsequent calls to the function `addCoveredStatistic`.
         *
         * This function must be invoked each time an existing rule has been refined, i.e., when a new condition has
         * been added to its body, because this results in a subset of the statistics being covered by the refined rule.
         *
         * This function is supposed to reset any non-global internal state that only holds for a certain subset of the
         * available statistics and therefore becomes invalid when a different subset of the statistics should be used.
         */
        virtual void resetCoveredStatistics() = 0;

        /**
         * Adds a specific statistic to the subset that is covered by an existing rule and therefore should be
         * considered in the following for refining an existing rule.
         *
         * This function must be called repeatedly for each statistic that is covered by the existing rule, immediately
         * after the invocation of the function `resetCoveredStatistics`.
         *
         * This function is supposed to update any internal state that relates to the considered statistics, i.e., to
         * compute and store local information that is required by the other functions that will be called later. Any
         * information computed by this function is expected to be reset when invoking the function
         * `resetCoveredStatistics` for the next time.
         *
         * @param statisticIndex The index of the statistic that should be added
         */
        virtual void addCoveredStatistic(uint32 statisticIndex) = 0;

        /**
         * Removes a specific statistic from the subset that is covered by an existing rule and therefore should not be
         * considered in the following for refining an existing rule.
         *
         * This function must be called repeatedly for each statistic that is not covered anymore by the existing rule.
         *
         * This function is supposed to update any internal state that relates to the considered statistics, i.e., to
         * compute and store local information that is required by the other functions that will be called later. Any
         * information computed by this function is expected to be reset when invoking the function
         * `resetCoveredStatistics` for the next time.
         *
         * @param statisticIndex The index of the statistic that should be removed
         */
        virtual void removeCoveredStatistic(uint32 statisticIndex) = 0;

        /**
         * Creates and returns a new object of type `IResettableStatisticsSubset` that includes only those outputs,
         * whose indices are provided by a specific `CompleteIndexVector`.
         *
         * @param excludedStatisticIndices  A reference to an object of type `BinaryDokVector` that provides access to
         *                                  the indices of the statistics that should be excluded from the subset
         * @param outputIndices             A reference to an object of type `CompleteIndexVector` that provides access
         *                                  to the indices of the outputs that should be included in the subset
         * @return                          An unique pointer to an object of type `IResettableStatisticsSubset` that
         *                                  has been created
         */
        virtual std::unique_ptr<IResettableStatisticsSubset> createSubset(
          const BinaryDokVector& excludedStatisticIndices, const CompleteIndexVector& outputIndices) const = 0;

        /**
         * Creates and returns a new object of type `IResettableStatisticsSubset` that includes only those outputs,
         * whose indices are provided by a specific `PartialIndexVector`.
         *
         * @param excludedStatisticIndices  A reference to an object of type `BinaryDokVector` that provides access to
         *                                  the indices of the statistics that should be excluded from the subset
         * @param outputIndices             A reference to an object of type `PartialIndexVector` that provides access
         *                                  to the indices of the outputs that should be included in the subset
         * @return                          An unique pointer to an object of type `IResettableStatisticsSubset` that
         *                                  has been created
         */
        virtual std::unique_ptr<IResettableStatisticsSubset> createSubset(
          const BinaryDokVector& excludedStatisticIndices, const PartialIndexVector& outputIndices) const = 0;
};

/**
 * An abstract base class for all classes that provide access to statistics.
 *
 * @tparam State            The type of the state of the training process
 * @tparam StatisticVector  The type of the vectors that are used to store statistics
 * @tparam WeightVector     The type of the vector that provides access to the weights of individual statistics
 */
template<typename State, typename StatisticVector, typename WeightVector>
class AbstractWeightedStatistics : public AbstractStatisticsSpace<State>,
                                   virtual public IWeightedStatistics {
    private:

        template<typename StatisticView>
        static inline void addStatisticInternally(const EqualWeightVector& weights, const StatisticView& statisticView,
                                                  StatisticVector& statisticVector, uint32 statisticIndex) {
            statisticVector.add(statisticView, statisticIndex);
        }

        template<typename Weights, typename StatisticView>
        static inline void addStatisticInternally(const Weights& weights, const StatisticView& statisticView,
                                                  StatisticVector& statisticVector, uint32 statisticIndex) {
            typename Weights::weight_type weight = weights[statisticIndex];
            statisticVector.add(statisticView, statisticIndex, weight);
        }

        template<typename StatisticView>
        static inline void removeStatisticInternally(const EqualWeightVector& weights,
                                                     const StatisticView& statisticView,
                                                     StatisticVector& statisticVector, uint32 statisticIndex) {
            statisticVector.remove(statisticView, statisticIndex);
        }

        template<typename Weights, typename StatisticView>
        static inline void removeStatisticInternally(const Weights& weights, const StatisticView& statisticView,
                                                     StatisticVector& statisticVector, uint32 statisticIndex) {
            typename Weights::weight_type weight = weights[statisticIndex];
            statisticVector.remove(statisticView, statisticIndex, weight);
        }

    protected:

        /**
         * Initializes a given vector by setting its element for each output to the weighted sum of the statistics in a
         * specific view.
         *
         * @tparam StatisticView    The type of the view that provides access to the statistics
         * @param weights           A reference to an object of template type `WeightVector` that provides access to the
         *                          weights of individual statistics
         * @param statisticView     A reference to an object of template type `StatisticView` that provides access to
         *                          the statistics
         * @param statisticVector   A reference to an object of template type `StatisticVector` to be initialized
         */
        template<typename StatisticView>
        static inline void initializeSumVector(const WeightVector& weights, const StatisticView& statisticView,
                                               StatisticVector& statisticVector) {
            uint32 numStatistics = weights.getNumElements();

            for (uint32 i = 0; i < numStatistics; i++) {
                addStatisticInternally(weights, statisticView, statisticVector, i);
            }
        }

        /**
         * A reference to an object of template type `WeightVector` that provides access to the weights of individual
         * statistics.
         */
        const WeightVector& weights_;

        /**
         * A reference to an object of template type `WeightVector` that provides access to the weights of individual
         * statistics.
         */
        StatisticVector totalSumVector_;

    public:

        /**
         * @param state   A reference to an object of template type `State` that represents the state of the training
         *                process
         * @param weights A reference to an object of template type `WeightVector` that provides access to the weights
         *                of individual statistics
         */
        AbstractWeightedStatistics(State& state, const WeightVector& weights)
            : AbstractStatisticsSpace<State>(state), weights_(weights),
              totalSumVector_(state.statisticMatrixPtr->getNumCols(), true) {
            initializeSumVector(weights, state.statisticMatrixPtr->getView(), totalSumVector_);
        }

        /**
         * @param other A reference to an object of type `AbstractWeightedStatistics` to be copied
         */
        AbstractWeightedStatistics(const AbstractWeightedStatistics<State, StatisticVector, WeightVector>& other)
            : AbstractStatisticsSpace<State>(other.state_), weights_(other.weights_),
              totalSumVector_(other.totalSumVector_) {}

        virtual ~AbstractWeightedStatistics() override {}

        /**
         * @see `IWeightedStatistics::resetCoveredStatistics`
         */
        void resetCoveredStatistics() override {
            totalSumVector_.clear();
        }

        /**
         * @see `IWeightedStatistics::addCoveredStatistic`
         */
        void addCoveredStatistic(uint32 statisticIndex) override {
            addStatisticInternally(weights_, this->state_.statisticMatrixPtr->getView(), totalSumVector_,
                                   statisticIndex);
        }

        /**
         * @see `IWeightedStatistics::removeCoveredStatistic`
         */
        void removeCoveredStatistic(uint32 statisticIndex) override {
            removeStatisticInternally(weights_, this->state_.statisticMatrixPtr->getView(), totalSumVector_,
                                      statisticIndex);
        }
};
