/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/simd/functions/difference.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template<typename Arch, typename T>
    void difference(Arch, T* a, const T* b, const T* c, uint32 numElements) {
        typedef xsimd::batch<T, Arch> batch;
        constexpr std::size_t batchSize = batch::size;
        uint32 batchEnd = numElements - (numElements % batchSize);
        uint32 i = 0;

        for (; i < batchEnd; i += batchSize) {
            batch batchB = batch::load_unaligned(b + i);
            batch batchC = batch::load_unaligned(c + i);
            (batchB - batchC).store_unaligned(a + i);
        }

        for (; i < numElements; i++) {
            a[i] = b[i] - c[i];
        }
    }
}
#endif
