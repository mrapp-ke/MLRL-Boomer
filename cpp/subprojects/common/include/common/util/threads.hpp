/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"
#include <thread>


/**
 * Returns the maximum number of threads that can be for parallelized algorithms.
 *
 * @param preferredNumThreads   The preferred number of threads or 0, if all available CPU cores should be used
 * @return                      The maximum number of threads that can be used
 */
static inline uint32 getMaxThreads(uint32 preferredNumThreads) {
    uint32 maxThreads = std::thread::hardware_concurrency();

    if (maxThreads == 0) {
        maxThreads = 1;
    }

    return preferredNumThreads > 0 ? std::min(maxThreads, preferredNumThreads) : maxThreads;
}
