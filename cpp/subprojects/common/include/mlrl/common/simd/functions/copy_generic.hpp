/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/simd/functions/copy.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template<typename Arch, typename T>
    void copy(Arch, const T* from, T* to, uint32 numElements) {
        using batch = xsimd::batch<T, Arch>;
        constexpr std::size_t batchSize = batch::size;
        const uint32 numBatches = numElements / batchSize;

        for (uint32 i = 0; i < numBatches; i++) {
            const uint32 offset = batchSize * i;
            batch batchFrom = util::load_simd<batch, const T>(from + offset);
            util::store_simd(batchFrom, to + offset);
        }
    }
}
#endif
