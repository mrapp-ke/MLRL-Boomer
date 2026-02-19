#include "mlrl/common/simd/functions/difference_generic.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template void difference<xsimd::sse2, float32>(xsimd::sse2, float32*, const float32*, const float32*, uint32);
    template void difference<xsimd::sse2, float64>(xsimd::sse2, float64*, const float64*, const float64*, uint32);
}
#endif
