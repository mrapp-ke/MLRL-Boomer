#include "mlrl/common/simd/functions/add_weighted_generic.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template void addWeighted<xsimd::neon64, float32>(xsimd::neon64, float32*, const float32*, uint32, float32);
    template void addWeighted<xsimd::neon64, float64>(xsimd::neon64, float64*, const float64*, uint32, float64);
}
#endif
