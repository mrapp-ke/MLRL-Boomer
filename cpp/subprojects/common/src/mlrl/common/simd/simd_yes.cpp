#include "mlrl/common/simd/simd_yes.hpp"

#include "mlrl/common/util/xsimd.hpp"

bool SimdConfig::isSimdEnabled(uint32 expectedBatchSize) const {
#if SIMD_SUPPORT_ENABLED
    return expectedBatchSize > 1 && !util::getSupportedSimdExtensions().empty();
#else
    return false;
#endif
}
