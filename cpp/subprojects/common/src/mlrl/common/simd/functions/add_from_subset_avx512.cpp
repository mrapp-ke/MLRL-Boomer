#include "mlrl/common/simd/functions/add_from_subset_gather.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template void addFromSubset<xsimd::avx512f, float32>(xsimd::avx512f, float32*, const float32*, const uint32*,
                                                         uint32);
    template void addFromSubset<xsimd::avx512f, float64>(xsimd::avx512f, float64*, const float64*, const uint32*,
                                                         uint32);
}
#endif
