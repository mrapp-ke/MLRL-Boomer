/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

/**
 * Defines an interface for all classes that allow to configure whether single instruction, multiple data (SIMD)
 * operations should be used by an algorithm.
 */
class ISimdConfig {
    public:

        virtual ~ISimdConfig() {}

        /**
         * Returns whether SIMD operations should be used or not.
         *
         * @param expectedBatchSize The typical batch size that is expected
         * @return                  True, if SIMD operations should be used, false otherwise
         */
        virtual bool isSimdEnabled(uint32 expectedBatchSize) const = 0;
};
