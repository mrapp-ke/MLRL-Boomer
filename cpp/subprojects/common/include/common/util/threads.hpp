/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"
#include <thread>


/**
 * Returns the number of threads to be used for parallelized algorithms.
 *
 * @param preferredNumThreads   The preferred number of threads or 0, if all available CPU cores should be used
 * @return                      The number of threads to be used
 */
static inline uint32 getNumThreads(uint32 preferredNumThreads) {
    uint32 numThreads = std::thread::hardware_concurrency();
    return numThreads != 0 ? numThreads : 1;
}
