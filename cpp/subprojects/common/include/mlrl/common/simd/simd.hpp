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
         * Returns whether SIMD operations should be used, depending on the expected batch size, or not.
         *
         * @param expectedBatchSize The typical batch size that is expected
         * @return                  True, if SIMD operations should be used, false otherwise
         */
        virtual bool isSimdRecommended(uint32 expectedBatchSize) const = 0;

        /**
         * Returns whether SIMD operations can be used or not.
         *
         * @return True, if SIMD operations can be used, false otherwise
         */
        virtual bool isSimdEnabled() const = 0;
};
