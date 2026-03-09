/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/simd/simd.hpp"

/**
 * Allows to configure that single instruction, multiple data (SIMD) operations should be used by an algorithm.
 */
class SimdConfig final : public ISimdConfig {
    public:

        bool isSimdEnabled() const override;
};
