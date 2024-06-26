/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/statistics/statistics_weighted_immutable.hpp"

#include <memory>

/**
 * Defines an interface for all classes that inherit from `IImmutableWeightedStatistics`, but do also provide functions
 * that allow to only use a sub-sample of the available statistics.
 */
class IWeightedStatistics : virtual public IImmutableWeightedStatistics {
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
};
