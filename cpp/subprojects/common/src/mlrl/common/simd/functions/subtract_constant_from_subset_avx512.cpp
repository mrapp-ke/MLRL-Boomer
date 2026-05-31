#include "mlrl/common/simd/functions/subtract_constant_from_subset_gather.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template void subtractConstantFromSubset<xsimd::avx512f, uint32>(xsimd::avx512f, uint32*, uint32, const uint32*,
                                                                     uint32);
    template void subtractConstantFromSubset<xsimd::avx512f, float32>(xsimd::avx512f, float32*, float32, const uint32*,
                                                                      uint32);
    template void subtractConstantFromSubset<xsimd::avx512f, float64>(xsimd::avx512f, float64*, float64, const uint32*,
                                                                      uint32);
}
#endif
