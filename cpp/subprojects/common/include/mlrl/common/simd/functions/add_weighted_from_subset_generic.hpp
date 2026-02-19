/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/simd/functions/add_weighted_from_subset.hpp"

#include <array>

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template<typename Arch, typename T>
    void addWeightedFromSubset(Arch, T* a, const T* b, const uint32* indices, uint32 numElements, T weight) {
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

            batch batchA = batch::load_unaligned(a + i);
            batch batchTmp = batch::load_unaligned(tmp.data());
            (batchA + (batchTmp * weight)).store_unaligned(a + i);
        }

        for (; i < numElements; i++) {
            uint32 index = indices[i];
            a[i] += (b[index] * weight);
        }
    }
}
#endif
