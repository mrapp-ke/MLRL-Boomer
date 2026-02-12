#include "mlrl/common/simd/functions/copy.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template void copy<xsimd::avx512f, float32>(xsimd::avx512f, const float32*, float32*, uint32);
    template void copy<xsimd::avx512f, float64>(xsimd::avx512f, const float64*, float64*, uint32);
}
#endif
