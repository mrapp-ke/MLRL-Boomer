#include "mlrl/common/simd/functions/difference_with_subset_generic.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template void differenceWithSubset<xsimd::avx, float32>(xsimd::avx, float32*, const float32*, const float32*,
                                                            const uint32*, uint32);
    template void differenceWithSubset<xsimd::avx, float64>(xsimd::avx, float64*, const float64*, const float64*,
                                                            const uint32*, uint32);
}
#endif
