#include "mlrl/common/simd/functions/subtract_generic.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template void subtract<xsimd::neon64, float32>(xsimd::neon64, float32*, const float32*, uint32);
    template void subtract<xsimd::neon64, float64>(xsimd::neon64, float64*, const float64*, uint32);
}
#endif
