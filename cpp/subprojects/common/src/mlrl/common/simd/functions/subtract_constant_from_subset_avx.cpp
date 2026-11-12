#include "mlrl/common/simd/functions/subtract_constant_from_subset_generic.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template void subtractConstantFromSubset<xsimd::avx, uint32>(xsimd::avx, uint32*, uint32, const uint32*, uint32);
    template void subtractConstantFromSubset<xsimd::avx, float32>(xsimd::avx, float32*, float32, const uint32*, uint32);
    template void subtractConstantFromSubset<xsimd::avx, float64>(xsimd::avx, float64*, float64, const uint32*, uint32);
}
#endif
