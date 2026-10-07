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
        uint32 batchEnd = numIndices - (numIndices % batchSize);
        std::array<T, batchSize> tmp;
        uint32 i = 0;

        for (; i < batchEnd; i += batchSize) {
            for (uint32 j = 0; j < batchSize; j++) {
                uint32 index = indices[i + j];
                tmp[j] = array[index];
            }

            batch batchTmp = util::load_simd<batch, T>(tmp.data());
            (batchTmp - constant).store_unaligned(tmp.data());

            for (uint32 j = 0; j < batchSize; j++) {
                uint32 index = indices[i + j];
                array[index] = tmp[j];
            }
        }

        for (; i < numIndices; i++) {
            uint32 index = indices[i];
            array[index] -= constant;
        }
    }
}
#endif
