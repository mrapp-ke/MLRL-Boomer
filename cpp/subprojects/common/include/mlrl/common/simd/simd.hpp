/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

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
         * @return True, if SIMD operations should be used, false otherwise
         */
        virtual bool isSimdEnabled() const = 0;
};
