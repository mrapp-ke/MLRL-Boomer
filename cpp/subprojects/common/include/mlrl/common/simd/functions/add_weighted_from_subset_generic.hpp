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
        const uint32 numBatches = numElements / batchSize;
        alignas(Arch::alignment()) std::array<T, batchSize> tmp;

        for (uint32 i = 0; i < numBatches; i++) {
            const uint32 offset = batchSize * i;

            for (std::size_t j = 0; j < batchSize; j++) {
                uint32 index = indices[offset + j];
                tmp[j] = b[index];
            }

            batch batchA = util::load_simd<batch, T>(a + offset);
            batch batchTmp = util::load_simd<batch, T>(tmp.data());
            util::store_simd(batchA + (batchTmp * weight), a + offset);
        }
    }
}
#endif
