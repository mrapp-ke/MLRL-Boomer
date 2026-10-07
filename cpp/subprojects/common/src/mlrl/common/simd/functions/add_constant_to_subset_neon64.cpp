#include "mlrl/common/simd/functions/add_constant_to_subset_generic.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template void addConstantToSubset<xsimd::neon64, uint32>(xsimd::neon64, uint32*, uint32, const uint32*, uint32);
    template void addConstantToSubset<xsimd::neon64, float32>(xsimd::neon64, float32*, float32, const uint32*, uint32);
    template void addConstantToSubset<xsimd::neon64, float64>(xsimd::neon64, float64*, float64, const uint32*, uint32);
}
#endif
