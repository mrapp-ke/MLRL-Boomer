/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/statistics/quantization.hpp"

#include <memory>

namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure a method for quantizing statistics that does not
     * actually perform any quantization.
     */
    class NoQuantizationConfig final : public IQuantizationConfig {
        public:

            std::unique_ptr<IQuantizationFactory> createQuantizationFactory() const override;
    };

}
