#include "mlrl/common/simd/functions/add_constant_to_subset_gather.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template void addConstantToSubset<xsimd::avx512f, uint32>(xsimd::avx512f, uint32*, uint32, const uint32*, uint32);
    template void addConstantToSubset<xsimd::avx512f, float32>(xsimd::avx512f, float32*, float32, const uint32*,
                                                               uint32);
    template void addConstantToSubset<xsimd::avx512f, float64>(xsimd::avx512f, float64*, float64, const uint32*,
                                                               uint32);
}
#endif
