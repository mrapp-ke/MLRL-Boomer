/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"

#include <algorithm>
#include <thread>

/**
 * Returns the number of threads that are available for parallelized algorithms. If multi-threading support is disabled,
 * the number of available threads is limited to 1.
 *
 * @param numPreferredThreads   The preferred number of threads. Must be at least 1 or 0, if all available CPU cores
 *                              should be utilized
 * @return                      The number of available threads
 */
static inline uint32 getNumAvailableThreads(uint32 numPreferredThreads) {
#if MULTI_THREADING_SUPPORT_ENABLED
    uint32 numAvailableThreads = std::max<uint32>(std::thread::hardware_concurrency(), 1);
    return numPreferredThreads > 0 ? std::min(numAvailableThreads, numPreferredThreads) : numAvailableThreads;
#else
    return 1;
#endif
}
