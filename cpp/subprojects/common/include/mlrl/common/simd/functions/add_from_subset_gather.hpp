/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/simd/functions/add_from_subset.hpp"

#include <type_traits>

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template<typename Arch, typename T>
    void addFromSubset(Arch, T* a, const T* b, const uint32* indices, uint32 numElements) {
        using batch = xsimd::batch<T, Arch>;
        using index_type =
          std::conditional_t<xsimd::batch<T, Arch>::size == xsimd::batch<uint32, Arch>::size, uint32, uint64>;
        using index_batch = xsimd::batch<index_type, Arch>;
        constexpr std::size_t batchSize = batch::size;
        uint32 batchEnd = numElements - (numElements % batchSize);
        uint32 i = 0;

        for (; i < batchEnd; i += batchSize) {
            batch batchA = util::load_simd<batch, T>(a + i);
            index_batch batchIndices = util::load_simd<index_batch, const uint32>(indices + i);
            batch batchB = batch::gather(b, batchIndices);
            (batchA + batchB).store_unaligned(a + i);
        }

        for (; i < numElements; i++) {
            uint32 index = indices[i];
            a[i] += b[index];
        }
    }
}
#endif
