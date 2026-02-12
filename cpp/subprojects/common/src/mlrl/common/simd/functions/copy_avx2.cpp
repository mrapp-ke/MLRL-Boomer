#include "mlrl/common/simd/functions/copy.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template void copy<xsimd::avx2, float32>(xsimd::avx2, const float32*, float32*, uint32);
    template void copy<xsimd::avx2, float64>(xsimd::avx2, const float64*, float64*, uint32);
}
#endif
