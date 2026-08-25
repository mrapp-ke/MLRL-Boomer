#include "mlrl/common/simd/functions/add_from_subset_generic.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template void addFromSubset<xsimd::neon64, float32>(xsimd::neon64, float32*, const float32*, const uint32*, uint32);
    template void addFromSubset<xsimd::neon64, float64>(xsimd::neon64, float64*, const float64*, const uint32*, uint32);
}
#endif
