#include "mlrl/common/simd/functions/add_constant_to_subset_generic.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template void addConstantToSubset<xsimd::sse2, uint32>(xsimd::sse2, uint32*, uint32, const uint32*, uint32);
    template void addConstantToSubset<xsimd::sse2, float32>(xsimd::sse2, float32*, float32, const uint32*, uint32);
    template void addConstantToSubset<xsimd::sse2, float64>(xsimd::sse2, float64*, float64, const uint32*, uint32);
}
#endif
