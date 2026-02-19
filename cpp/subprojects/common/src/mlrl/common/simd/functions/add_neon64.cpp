#include "mlrl/common/simd/functions/add_generic.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template void add<xsimd::neon64, float32>(xsimd::neon64, float32*, const float32*, uint32);
    template void add<xsimd::neon64, float64>(xsimd::neon64, float64*, const float64*, uint32);
}
#endif
