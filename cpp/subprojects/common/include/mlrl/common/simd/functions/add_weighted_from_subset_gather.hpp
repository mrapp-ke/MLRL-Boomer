/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/simd/functions/add_weighted_from_subset.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template<typename Arch, typename T>
    void addWeightedFromSubset(Arch, T* a, const T* b, const uint32* indices, uint32 numElements, T weight) {
        using batch = xsimd::batch<T, Arch>;
        using index_type =
          std::conditional_t<xsimd::batch<T, Arch>::size == xsimd::batch<uint32, Arch>::size, uint32, uint64>;
        using index_batch = xsimd::batch<index_type, Arch>;
        constexpr std::size_t batchSize = batch::size;
        const uint32 numBatches = numElements / batchSize;

        for (uint32 i = 0; i < numBatches; i++) {
            const uint32 offset = batchSize * i;
            batch batchA = util::load_simd<batch, T>(a + offset);
            index_batch batchIndices = util::load_simd<index_batch, const uint32>(indices + offset);
            batch batchB = batch::gather(b, batchIndices);
            util::store_simd(batchA + (batchB * weight), a + offset);
        }
    }
}
#endif
