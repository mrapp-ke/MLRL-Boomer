#include "mlrl/common/simd/functions/difference.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template void difference<xsimd::neon64, float32>(xsimd::neon64, float32*, const float32*, const float32*, uint32);
    template void difference<xsimd::neon64, float64>(xsimd::neon64, float64*, const float64*, const float64*, uint32);
}
#endif
