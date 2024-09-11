/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

/**
 * Defines an interface for all classes that provide access to the statistics space.
 */
class IStatisticsSpace {
    public:

        virtual ~IStatisticsSpace() {}

        /**
         * Returns the number of statistics.
         *
         * @return The number of statistics
         */
        virtual uint32 getNumStatistics() const = 0;

        /**
         * Returns the number of outputs.
         *
         * @return The number of outputs
         */
        virtual uint32 getNumOutputs() const = 0;
};
