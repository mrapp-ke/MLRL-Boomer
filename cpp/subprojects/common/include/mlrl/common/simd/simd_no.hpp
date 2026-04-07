/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/simd/simd.hpp"

/**
 * Allows to configure that no single instruction, multiple data (SIMD) operations should be used by an algorithm.
 */
class NoSimdConfig final : public ISimdConfig {
    public:

        bool isSimdEnabled() const override;
};
