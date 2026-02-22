#include "mlrl/common/simd/simd_yes.hpp"

#include "mlrl/common/util/xsimd.hpp"

bool SimdConfig::isSimdEnabled() const {
#if SIMD_SUPPORT_ENABLED
    return !util::getSupportedSimdExtensions().empty();
#else
    return false;
#endif
}
