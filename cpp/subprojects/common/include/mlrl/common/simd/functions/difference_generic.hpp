/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/simd/functions/difference.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template<typename Arch, typename T>
    void difference(Arch, T* a, const T* b, const T* c, uint32 numElements) {
        using batch = xsimd::batch<T, Arch>;
        constexpr std::size_t batchSize = batch::size;
        const uint32 numBatches = numElements / batchSize;

        for (uint32 i = 0; i < numBatches; i++) {
            const uint32 offset = batchSize * i;
            batch batchB = util::load_simd<batch, const T>(b + offset);
            batch batchC = util::load_simd<batch, const T>(c + offset);
            util::store_simd(batchB - batchC, a + offset);
        }
    }
}
#endif
