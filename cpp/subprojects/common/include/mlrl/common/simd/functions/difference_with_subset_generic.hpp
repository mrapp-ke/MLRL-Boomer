/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/simd/functions/difference_with_subset.hpp"

#include <array>

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template<typename Arch, typename T>
    void differenceWithSubset(Arch, T* a, const T* b, const T* c, const uint32* indices, uint32 numElements) {
        typedef xsimd::batch<T, Arch> batch;
        constexpr std::size_t batchSize = batch::size;
        uint32 batchEnd = numElements - (numElements % batchSize);
        std::array<T, batchSize> tmp;
        uint32 i = 0;

        for (; i < batchEnd; i += batchSize) {
            for (std::size_t j = 0; j < batchSize; j++) {
                uint32 index = indices[i + j];
                tmp[j] = b[index];
            }

            batch batchTmp = batch::load_unaligned(tmp.data());
            batch batchC = batch::load_unaligned(c + i);
            (batchTmp - batchC).store_unaligned(a + i);
        }

        for (; i < numElements; i++) {
            uint32 index = indices[i];
            a[i] = b[index] - c[i];
        }
    }
}
#endif
