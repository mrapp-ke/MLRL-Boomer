#include "mlrl/common/simd/simd_yes.hpp"

#include "mlrl/common/util/xsimd.hpp"

bool SimdConfig::isSimdRecommended(uint32 expectedBatchSize) const {
    return expectedBatchSize > 1 && SimdConfig::isSimdEnabled();
}

bool SimdConfig::isSimdEnabled() const {
#if SIMD_SUPPORT_ENABLED
    return !util::getSupportedSimdExtensions().empty();
#else
    return false;
#endif
}
