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

            batch batchTmp = util::load_simd<batch, T>(tmp.data());
            batch batchC = util::load_simd<batch, const T>(c + offset);
            util::store_simd(batchTmp - batchC, a + offset);
        }
    }
}
#endif
