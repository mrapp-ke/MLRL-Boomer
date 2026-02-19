/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/simd/functions/copy.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template<typename Arch, typename T>
    void copy(Arch, const T* from, T* to, uint32 numElements) {
        typedef xsimd::batch<T, Arch> batch;
        constexpr std::size_t batchSize = batch::size;
        uint32 batchEnd = numElements - (numElements % batchSize);
        uint32 i = 0;

        for (; i < batchEnd; i += batchSize) {
            batch::load_unaligned(from + i).store_unaligned(to + i);
        }

        for (; i < numElements; i++) {
            to[i] = from[i];
        }
    }
}
#endif
