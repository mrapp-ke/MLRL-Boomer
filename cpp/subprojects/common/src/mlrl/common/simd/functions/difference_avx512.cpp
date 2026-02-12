#include "mlrl/common/simd/functions/difference.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template void difference<xsimd::avx512f, float32>(xsimd::avx512f, float32*, const float32*, const float32*, uint32);
    template void difference<xsimd::avx512f, float64>(xsimd::avx512f, float64*, const float64*, const float64*, uint32);
}
#endif
