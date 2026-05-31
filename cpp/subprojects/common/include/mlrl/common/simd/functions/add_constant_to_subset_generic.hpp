/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/simd/functions/add_constant_to_subset.hpp"

#include <array>

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template<typename Arch, typename T>
    void addConstantToSubset(Arch, T* array, T constant, const uint32* indices, uint32 numIndices) {
        using batch = xsimd::batch<T, Arch>;
        constexpr std::size_t batchSize = batch::size;
        uint32 batchEnd = numIndices - (numIndices % batchSize);
        std::array<T, batchSize> tmp;
        uint32 i = 0;

        for (; i < batchEnd; i += batchSize) {
            for (std::size_t j = 0; j < batchSize; j++) {
                tmp[j] = array[indices[i + j]];
            }

            batch batchTmp = batch::load_unaligned(tmp.data());
            (batchTmp + constant).store_unaligned(tmp.data());

            for (std::size_t j = 0; j < batchSize; j++) {
                array[indices[i + j]] = tmp[j];
            }
        }

        for (; i < numIndices; i++) {
            array[indices[i]] += constant;
        }
    }
}
#endif
