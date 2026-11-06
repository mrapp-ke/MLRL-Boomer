#include "mlrl/common/simd/functions/subtract_weighted_generic.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template void subtractWeighted<xsimd::sse2, float32>(xsimd::sse2, float32*, const float32*, uint32, float32);
    template void subtractWeighted<xsimd::sse2, float64>(xsimd::sse2, float64*, const float64*, uint32, float64);
}
#endif
