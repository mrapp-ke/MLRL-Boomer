/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/simd/functions/add_weighted.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template<typename Arch, typename T>
    void addWeighted(Arch, T* a, const T* b, uint32 numElements, T weight) {
        typedef xsimd::batch<T, Arch> batch;
        constexpr std::size_t batchSize = batch::size;
        uint32 batchEnd = numElements - (numElements % batchSize);
        uint32 i = 0;

        for (; i < batchEnd; i += batchSize) {
            batch batchA = batch::load_unaligned(a + i);
            batch batchB = batch::load_unaligned(b + i);
            (batchA + (batchB * weight)).store_unaligned(a + i);
        }

        for (; i < numElements; i++) {
            a[i] += (b[i] * weight);
        }
    }
}
#endif
