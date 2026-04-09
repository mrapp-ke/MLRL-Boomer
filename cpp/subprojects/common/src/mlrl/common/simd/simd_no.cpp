#include "mlrl/common/simd/simd_no.hpp"

bool NoSimdConfig::isSimdRecommended(uint32 expectedbatchSize) const {
    return false;
}

bool NoSimdConfig::isSimdEnabled() const {
    return false;
}
