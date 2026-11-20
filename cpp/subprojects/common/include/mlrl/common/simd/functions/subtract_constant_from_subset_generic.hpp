/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/simd/functions/subtract_constant_from_subset.hpp"

#include <array>

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template<typename Arch, typename T>
    void subtractConstantFromSubset(Arch, T* array, T constant, const uint32* indices, uint32 numIndices) {
        using batch = xsimd::batch<T, Arch>;
        constexpr std::size_t batchSize = batch::size;
        const uint32 numBatches = numIndices / batchSize;
        alignas(Arch::alignment()) std::array<T, batchSize> tmp;

        for (uint32 i = 0; i < numBatches; i++) {
            const uint32 offset = batchSize * i;

            for (uint32 j = 0; j < batchSize; j++) {
                uint32 index = indices[offset + j];
                tmp[j] = array[index];
            }

            batch batchTmp = util::load_simd<batch, T>(tmp.data());
            util::store_simd(batchTmp - constant, tmp.data());

            for (uint32 j = 0; j < batchSize; j++) {
                uint32 index = indices[offset + j];
                array[index] = tmp[j];
            }
        }
    }
}
#endif
