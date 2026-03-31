#include "mlrl/common/simd/functions/difference_with_subset_gather.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template void differenceWithSubset<xsimd::avx2, float32>(xsimd::avx2, float32*, const float32*, const float32*,
                                                             const uint32*, uint32);
    template void differenceWithSubset<xsimd::avx2, float64>(xsimd::avx2, float64*, const float64*, const float64*,
                                                             const uint32*, uint32);
}
#endif
