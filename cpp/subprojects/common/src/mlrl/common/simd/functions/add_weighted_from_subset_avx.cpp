#include "mlrl/common/simd/functions/add_weighted_from_subset.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template void addWeightedFromSubset<xsimd::avx, float32>(xsimd::avx, float32*, const float32*, const uint32*,
                                                             uint32, float32);
    template void addWeightedFromSubset<xsimd::avx, float64>(xsimd::avx, float64*, const float64*, const uint32*,
                                                             uint32, float64);
}
#endif
