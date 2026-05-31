/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/simd/functions/subtract_constant_from_subset.hpp"

#include <type_traits>

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template<typename Arch, typename T>
    void subtractConstantFromSubset(Arch, T* array, T constant, const uint32* indices, uint32 numIndices) {
        using batch = xsimd::batch<T, Arch>;
        using index_type =
          std::conditional_t<xsimd::batch<T, Arch>::size == xsimd::batch<uint32, Arch>::size, uint32, uint64>;
        using index_batch = xsimd::batch<index_type, Arch>;
        constexpr std::size_t batchSize = batch::size;
        uint32 batchEnd = numIndices - (numIndices % batchSize);
        uint32 i = 0;

        for (; i < batchEnd; i += batchSize) {
            index_batch batchIndices = index_batch::load_unaligned(indices + i);
            batch batchArray = batch::gather(array, batchIndices);
            (batchArray - constant).scatter(array, batchIndices);
        }

        for (; i < numIndices; i++) {
            array[indices[i]] -= constant;
        }
    }
}
#endif
