#include "mlrl/common/simd/functions/subtract.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template void subtract<xsimd::avx512f, float32>(xsimd::avx512f, float32*, const float32*, uint32);
    template void subtract<xsimd::avx512f, float64>(xsimd::avx512f, float64*, const float64*, uint32);
}
#endif
