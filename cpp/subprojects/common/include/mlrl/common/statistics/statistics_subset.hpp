/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/sampling/weight_vector_equal.hpp"
#include "mlrl/common/statistics/statistics_update_candidate.hpp"

/**
 * Defines an interface for all classes that provide access to a subset of the statistics and allows to calculate the
 * scores to be predicted by rules that cover such a subset.
 */
class IStatisticsSubset {
    public:

        virtual ~IStatisticsSubset() {}

        /**
         * Returns whether the statistics at a specific index have a non-zero weight or not.
         *
         * @return True, if the statistics at the given index have a non-zero weight, false otherwise
         */
        virtual bool hasNonZeroWeight(uint32 statisticIndex) const = 0;

        /**
         * Adds the statistics at a specific index to the subset in order to mark it as covered by the condition that is
         * currently considered for refining a rule.
         *
         * This function must be called repeatedly for each statistic that is covered by the current condition,
         * immediately after the invocation of the function `IWeightedStatistics::createSubset`. If a rule has already
         * been refined, each of these statistics must have been marked as covered earlier via the function
         * `IWeightedStatistics::addCoveredStatistic` and must not have been marked as uncovered via the function
         * `IWeightedStatistics::removeCoveredStatistic`.
         *
         * This function is supposed to update any internal state of the subset that relates to the statistics that are
         * covered by the current condition, i.e., to compute and store local information that is required by the other
         * functions that will be called later. Any information computed by this function is expected to be reset when
         * invoking the function `resetSubset` for the next time.
         *
         * @param statisticIndex The index of the covered statistic
         */
        virtual void addToSubset(uint32 statisticIndex) = 0;

        /**
         * Calculates and returns the scores to be predicted by a rule that covers all statistics that have been added
         * to the subset via the function `addToSubset`, as well as a numerical score that assesses the overall quality
         * of the predicted scores.
         *
         * @return An unique pointer to an object of type `IStatisticsUpdateCandidate` that stores the scores to be
         *         predicted by the rule for each considered output, as well as a numerical score that assesses their
         *         overall quality
         */
        virtual std::unique_ptr<IStatisticsUpdateCandidate> calculateScores() = 0;
};

/**
 * An abstract base class for all classes that provide access to a subset of the statistics and allows to calculate the
 * scores to be predicted by rules that cover such a subset.
 *
 * @tparam State            The type of the state of the training process
 * @tparam StatisticVector  The type of the vector that is used to store the sums of statistics
 * @tparam WeightVector     The type of the vector that provides access to the weights of individual statistics
 * @tparam IndexVector      The type of the vector that provides access to the indices of the outputs that are included
 *                          in the subset
 */
template<typename State, typename StatisticVector, typename WeightVector, typename IndexVector>
class AbstractStatisticsSubset : virtual public IStatisticsSubset {
    private:

        static inline bool hasNonZeroWeightInternally(const EqualWeightVector& weights, uint32 statisticIndex) {
            return true;
        }

        template<typename Weights>
        static inline bool hasNonZeroWeightInternally(const Weights& weights, uint32 statisticIndex) {
            return !isEqualToZero(weights[statisticIndex]);
        }

        template<typename StatisticView>
        static inline void addStatisticToSubsetInternally(const EqualWeightVector& weights,
                                                          const StatisticView& statisticView, StatisticVector& vector,
                                                          const IndexVector& outputIndices, uint32 statisticIndex) {
            vector.addToSubset(statisticView, statisticIndex, outputIndices);
        }

        template<typename StatisticView, typename Weights>
        static inline void addStatisticToSubsetInternally(const Weights& weights, const StatisticView& statisticView,
                                                          StatisticVector& vector, const IndexVector& outputIndices,
                                                          uint32 statisticIndex) {
            typename WeightVector::weight_type weight = weights[statisticIndex];
            vector.addToSubset(statisticView, statisticIndex, outputIndices, weight);
        }

    protected:

        /**
         * A reference to an object of template type `State` that represents the state of the training process.
         */
        State& state_;

        /**
         * An object of template type `StatisticVector` that stores the sums of statistics.
         */
        StatisticVector sumVector_;

        /**
         * A reference to an object of template type `WeightVector` that provides access to the weights of individual
         * statistics.
         */
        const WeightVector& weights_;

        /**
         * A reference to an object of template type `IndexVector` that provides access to the indices of the outputs
         * that are included in the subset.
         */
        const IndexVector& outputIndices_;

    public:

        /**
         * @param state         A reference to an object of template type `State` that represents the state of the
         *                      training process
         * @param weights       A reference to an object of template type `WeightVector` that provides access to the
         *                      weights of individual statistics
         * @param outputIndices A reference to an object of template type `IndexVector` that provides access to the
         *                      indices of the outputs that are included in the subset
         */
        AbstractStatisticsSubset(State& state, const WeightVector& weights, const IndexVector& outputIndices)
            : state_(state), sumVector_(outputIndices.getNumElements(), true), weights_(weights),
              outputIndices_(outputIndices) {}

        virtual ~AbstractStatisticsSubset() override {}

        /**
         * @see `IStatisticsSubset::hasNonZeroWeight`
         */
        bool hasNonZeroWeight(uint32 statisticIndex) const override final {
            return this->hasNonZeroWeightInternally(weights_, statisticIndex);
        }

        /**
         * @see `IStatisticsSubset::addToSubset`
         */
        void addToSubset(uint32 statisticIndex) override final {
            this->addStatisticToSubsetInternally(weights_, state_.statisticMatrixPtr->getView(), sumVector_,
                                                 outputIndices_, statisticIndex);
        }
};

/**
 * Adds the statistics at a specific row of a view to a given vector. The statistics are weighted according to given
 * weights.
 *
 * @tparam StatisticVector  The type of the vector to be modified
 * @tparam StatisticView    The type of the view that provides access to the statistics
 * @param statisticVector   A reference to an object of template type `StatisticVector` to be modified
 * @param weights           A reference to an object of type `EqualWeightVector` that provides access to the weights
 * @param statisticView     A reference to an object of template type `StatisticView` that provides access to the
 *                          statistics
 * @param row               The index of the row in the view to be added to the vector
 */
template<typename StatisticVector, typename StatisticView>
static inline void addStatisticsToVector(StatisticVector& statisticVector, const EqualWeightVector& weights,
                                         const StatisticView& statisticView, uint32 row) {
    statisticVector.add(statisticView, row);
}

/**
 * Adds the statistics at a specific row of a view to a given vector. The statistics are weighted according to given
 * weights.
 *
 * @tparam StatisticVector  The type of the vector to be modified
 * @tparam WeightVector     The type of the vector that provides access to the weights of statistics
 * @tparam StatisticView    The type of the view that provides access to the statistics
 * @param statisticVector   A reference to an object of template type `StatisticVector` to be modified
 * @param weights           A reference to an object of template type `WeightVector` that provides access to the weights
 * @param statisticView     A reference to an object of template type `StatisticView` that provides access to the
 *                          statistics
 * @param row               The index of the row in the view to be added to the vector
 */
template<typename StatisticVector, typename WeightVector, typename StatisticView>
static inline void addStatisticsToVector(StatisticVector& statisticVector, const WeightVector& weights,
                                         const StatisticView& statisticView, uint32 row) {
    typename WeightVector::weight_type weight = weights[row];
    statisticVector.add(statisticView, row, weight);
}

/**
 * Removes the statistics at a specific row of a view from a given vector. The statistics are weighted according to
 * given weights.
 *
 * @tparam StatisticVector  The type of the vector to be modified
 * @tparam StatisticView    The type of the view that provides access to the statistics
 * @param statisticVector   A reference to an object of template type `StatisticVector` to be modified
 * @param weights           A reference to an object of type `EqualWeightVector` that provides access to the weights
 * @param statisticView     A reference to an object of template type `StatisticView` that provides access to the
 *                          statistics
 * @param row               The index of the row in the view to be removed from the vector
 */
template<typename StatisticVector, typename StatisticView>
static inline void removeStatisticsFromVector(StatisticVector& statisticVector, const EqualWeightVector& weights,
                                              const StatisticView& statisticView, uint32 row) {
    statisticVector.remove(statisticView, row);
}

/**
 * Removes the statistics at a specific row of a view from a given vector. The statistics are weighted according to
 * given weights.
 *
 * @tparam StatisticVector  The type of the vector to be modified
 * @tparam WeightVector     The type of the vector that provides access to the weights of statistics
 * @tparam StatisticView    The type of the view that provides access to the statistics
 * @param statisticVector   A reference to an object of template type `StatisticVector` to be modified
 * @param weights           A reference to an object of template type `WeightVector` that provides access to the weights
 * @param statisticView     A reference to an object of template type `StatisticView` that provides access to the
 *                          statistics
 * @param row               The index of the row in the view to be removed from the vector
 */
template<typename StatisticVector, typename WeightVector, typename StatisticView>
static inline void removeStatisticsFromVector(StatisticVector& statisticVector, const WeightVector& weights,
                                              const StatisticView& statisticView, uint32 row) {
    typename WeightVector::weight_type weight = weights[row];
    statisticVector.remove(statisticView, row, weight);
}

/**
 * Initializes a given vector by setting its element for each output to the weighted sum of the statistics in a specific
 * view.
 *
 * @tparam StatisticVector  The type of the vector to be initialized
 * @tparam WeightVector     The type of the vector that provides access to the weights of statistics
 * @tparam StatisticView    The type of the view that provides access to the statistics
 * @param statisticVector   A reference to an object of template type `StatisticVector` to be initialized
 * @param weights           A reference to an object of template type `WeightVector` that provides access to the weights
 * @param statisticView     A reference to an object of template type `StatisticView` that provides access to the
 *                          statistics
 */
template<typename StatisticVector, typename WeightVector, typename StatisticView>
static inline void setVectorToWeightedSumOfStatistics(StatisticVector& statisticVector, const WeightVector& weights,
                                                      const StatisticView& statisticView) {
    uint32 numStatistics = weights.getNumElements();

    for (uint32 i = 0; i < numStatistics; i++) {
        addStatisticsToVector(statisticVector, weights, statisticView, i);
    }
}
