/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/simd/functions/subtract_weighted.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template<typename Arch, typename T>
    void subtractWeighted(Arch, T* a, const T* b, uint32 numElements, T weight) {
        using batch = xsimd::batch<T, Arch>;
        constexpr std::size_t batchSize = batch::size;
        const uint32 numBatches = numElements / batchSize;

        for (uint32 i = 0; i < numBatches; i++) {
            const uint32 offset = batchSize * i;
            batch batchA = util::load_simd<batch, T>(a + offset);
            batch batchB = util::load_simd<batch, const T>(b + offset);
            util::store_simd(batchA - (batchB * weight), a + offset);
        }
    }
}
#endif
