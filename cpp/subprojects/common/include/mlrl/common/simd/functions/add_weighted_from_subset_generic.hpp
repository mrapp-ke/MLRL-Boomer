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
        using batch = xsimd::batch<T, Arch>;
        constexpr std::size_t batchSize = batch::size;
        uint32 batchEnd = numElements - (numElements % batchSize);
        alignas(Arch::alignment()) std::array<T, batchSize> tmp;
        uint32 i = 0;

        for (; i < batchEnd; i += batchSize) {
            for (std::size_t j = 0; j < batchSize; j++) {
                uint32 index = indices[i + j];
                tmp[j] = b[index];
            }

            batch batchA = util::load_simd<batch, T>(a + i);
            batch batchTmp = util::load_simd<batch, T>(tmp.data());
            util::store_simd(batchA + (batchTmp * weight), a + i);
        }

        for (; i < numElements; i++) {
            uint32 index = indices[i];
            a[i] += (b[index] * weight);
        }
    }
}
#endif
