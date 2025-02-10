/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

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
