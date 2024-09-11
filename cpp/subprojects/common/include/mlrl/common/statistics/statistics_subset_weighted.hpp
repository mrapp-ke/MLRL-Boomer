/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/statistics/statistics_subset.hpp"

/**
 * Defines an interface for all classes that provide access to a subset of the weighted statistics that are stored by an
 * instance of the class `IWeightedStatistics` and allows to calculate the scores to be predicted by rules that cover
 * such a subset.
 */
class IWeightedStatisticsSubset : virtual public IStatisticsSubset {
    public:

        virtual ~IWeightedStatisticsSubset() override {}

        /**
         * Resets the subset by removing all statistics that have been added via preceding calls to the function
         * `addToSubset`.
         *
         * This function is supposed to reset the internal state of the subset to the state when the subset was created
         * via the function `IWeightedStatistics::createSubset`. When calling this function, the current state must not
         * be purged entirely, but it must be cached and made available for use by the functions `evaluateAccumulated`
         * and `evaluateUncoveredAccumulated`.
         *
         * This function may be invoked multiple times with one or several calls to `addToSubset` in between, which is
         * supposed to update the previously cached state by accumulating the current one, i.e., the accumulated cached
         * state should be the same as if `resetSubset` would not have been called at all.
         */
        virtual void resetSubset() = 0;

        /**
         * Calculates and returns the scores to be predicted by a rule that covers all statistics that have been added
         * to the subset via the function `addToSubset`, as well as a numerical score that assesses the quality of the
         * predicted scores. All statistics that have been added since the subset was created via the function
         * `IWeightedStatistics::createSubset` are taken into account even if the function `resetSubset` was called
         * since then.
         *
         * @return A reference to an object of type `IScoreVector` that stores the scores to be predicted by the rule
         *         for each considered output, as well as a numerical score that assesses their overall quality
         */
        virtual const IScoreVector& calculateScoresAccumulated() = 0;

        /**
         * Calculates and returns the scores to be predicted by a rule that covers all statistics that correspond to the
         * difference between the statistics that have been added to the subset via the function `addToSubset` and those
         * that have been marked as covered via the function `IWeightedStatistics::addCoveredStatistic` or
         * `IWeightedStatistics::removeCoveredStatistic`, as well as a numerical score that assesses the quality of the
         * predicted scores.
         *
         * @return A reference to an object of type `IScoreVector` that stores the scores to be predicted by the rule
         *         for each considered output, as well as a numerical score that assesses their overall quality
         */
        virtual const IScoreVector& calculateScoresUncovered() = 0;

        /**
         * Calculates and returns the scores to be predicted by a rule that covers all statistics that correspond to the
         * difference between the statistics that have been added to the subset via the function `addToSubset` and those
         * that have been marked as covered via the function `IWeightedStatistics::addCoveredStatistic` or
         * `IWeightedStatistics::removeCoveredStatistic`, as well as a numerical score that assesses the quality of the
         * predicted scores. All statistics that have been added since the subset was created via the function
         * `IWeightedStatistics::createSubset` are taken into account even if the function `resetSubset` was called
         * since then.
         *
         * @return A reference to an object of type `IScoreVector` that stores the scores to be predicted by the rule
         *         for each considered output, as well as a numerical score that assesses their overall quality
         */
        virtual const IScoreVector& calculateScoresUncoveredAccumulated() = 0;
};
