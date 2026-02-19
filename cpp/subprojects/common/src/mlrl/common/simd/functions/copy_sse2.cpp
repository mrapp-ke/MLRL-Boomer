#include "mlrl/common/simd/functions/copy_generic.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template void copy<xsimd::sse2, float32>(xsimd::sse2, const float32*, float32*, uint32);
    template void copy<xsimd::sse2, float64>(xsimd::sse2, const float64*, float64*, uint32);
}
#endif
