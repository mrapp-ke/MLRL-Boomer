/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"


/**
 * Defines an interface for all classes that allow to configure the multi-threading behavior of a parallelizable
 * algorithm.
 */
class IMultiThreadingConfig {

    public:

        virtual ~IMultiThreadingConfig() { };

        /**
         * Determines and returns the number of threads to be used by a parallelizable algorithm.
         *
         * @return The number of threads to be used
         */
        virtual uint32 configure() const = 0;

};
