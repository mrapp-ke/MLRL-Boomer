#include "mlrl/common/simd/functions/copy_generic.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template void copy<xsimd::avx, float32>(xsimd::avx, const float32*, float32*, uint32);
    template void copy<xsimd::avx, float64>(xsimd::avx, const float64*, float64*, uint32);
}
#endif
