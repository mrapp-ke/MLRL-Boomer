/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/simd/functions/subtract_weighted.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template<typename Arch, typename T>
    void subtractWeighted(Arch, T* a, const T* b, uint32 numElements, T weight) {
        using batch = xsimd::batch<T, Arch>;
        constexpr std::size_t batchSize = batch::size;
        uint32 batchEnd = numElements - (numElements % batchSize);
        uint32 i = 0;

        for (; i < batchEnd; i += batchSize) {
            batch batchA = util::load_simd<batch, T>(a + i);
            batch batchB = util::load_simd<batch, const T>(b + i);
            (batchA - (batchB * weight)).store_unaligned(a + i);
        }

        for (; i < numElements; i++) {
            a[i] -= (b[i] * weight);
        }
    }
}
#endif
