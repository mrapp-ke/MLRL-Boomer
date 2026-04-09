#include "mlrl/common/simd/simd_no.hpp"

bool NoSimdConfig::isSimdEnabled(uint32 expectedbatchSize) const {
    return false;
}
