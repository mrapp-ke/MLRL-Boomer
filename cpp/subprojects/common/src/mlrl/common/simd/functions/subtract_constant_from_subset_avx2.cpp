#include "mlrl/common/simd/functions/subtract_constant_from_subset_generic.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template void subtractConstantFromSubset<xsimd::avx2, uint32>(xsimd::avx2, uint32*, uint32, const uint32*, uint32);
    template void subtractConstantFromSubset<xsimd::avx2, float32>(xsimd::avx2, float32*, float32, const uint32*,
                                                                   uint32);
    template void subtractConstantFromSubset<xsimd::avx2, float64>(xsimd::avx2, float64*, float64, const uint32*,
                                                                   uint32);
}
#endif
