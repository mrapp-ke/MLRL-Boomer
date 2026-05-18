/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/simd/functions/difference_with_subset.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template<typename Arch, typename T>
    void differenceWithSubset(Arch, T* a, const T* b, const T* c, const uint32* indices, uint32 numElements) {
        using batch = xsimd::batch<T, Arch>;
        using index_type =
          std::conditional_t<xsimd::batch<T, Arch>::size == xsimd::batch<uint32, Arch>::size, uint32, uint64>;
        using index_batch = xsimd::batch<index_type, Arch>;
        constexpr std::size_t batchSize = batch::size;
        uint32 batchEnd = numElements - (numElements % batchSize);
        uint32 i = 0;

        for (; i < batchEnd; i += batchSize) {
            index_batch batchIndices = index_batch::load_unaligned(indices + i);
            batch batchB = batch::gather(b, batchIndices);
            batch batchC = batch::load_unaligned(c + i);
            (batchB - batchC).store_unaligned(a + i);
        }

        for (; i < numElements; i++) {
            uint32 index = indices[i];
            a[i] = b[index] - c[i];
        }
    }
}
#endif
