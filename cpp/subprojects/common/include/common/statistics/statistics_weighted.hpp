/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/statistics/statistics_weighted_immutable.hpp"
#include "common/statistics/histogram.hpp"


/**
 * Defines an interface for all classes that inherit from `IImmutableWeightedStatistics`, but do also provide functions
 * that allow to only use a sub-sample of the available statistics.
 */
class IWeightedStatistics : virtual public IImmutableWeightedStatistics {

    public:

        virtual ~IWeightedStatistics() override { };

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
         * Alternatively, this function may be used to indicate that a statistic, which has previously been passed to
         * this function, should not be considered anymore by setting the argument `remove` accordingly.
         *
         * This function is supposed to update any internal state that relates to the considered statistics, i.e., to
         * compute and store local information that is required by the other functions that will be called later. Any
         * information computed by this function is expected to be reset when invoking the function
         * `resetCoveredStatistics` for the next time.
         *
         * @param statisticIndex    The index of the statistic that should be added
         * @param remove            False, if the statistic should be considered, True, if the statistic should not be
         *                          considered anymore
         */
        virtual void addCoveredStatistic(uint32 statisticIndex, bool remove) = 0;

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
         * @param statisticIndex    The index of the statistic that should be removed
         */
        virtual void removeCoveredStatistic(uint32 statisticIndex) = 0;

        /**
         * Creates and returns a new histogram based on the statistics.
         *
         * @return An unique pointer to an object of type `IHistogram` that has been created
         */
        virtual std::unique_ptr<IHistogram> createHistogram(uint32 numBins) const = 0;

};
