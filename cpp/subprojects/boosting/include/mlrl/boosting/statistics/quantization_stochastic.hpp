/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/statistics/quantization.hpp"
#include "mlrl/common/data/types.hpp"  // TODO Remove later

#include <memory>

namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure a method for quantizing statistics that uses a
     * stochastic rounding strategy.
     */
    class IStochasticQuantizationConfig {
        public:

            virtual ~IStochasticQuantizationConfig() {}

            /**
             * Returns the number of bits that are used for quantized statistics.
             *
             * @return The number of bits that are used
             */
            virtual uint32 getNumBits() const = 0;

            /**
             * Sets the number of bits to be used for quantized statistics.
             *
             * @param numBits   The number of bits to be used. Must be greater than 0
             * @return          A reference to an object of type `IStochasticQuantizationConfig` that allows further
             *                  configuration of the quantization method
             */
            virtual IStochasticQuantizationConfig& setNumBits(uint32 numBits) = 0;
    };

    /**
     * Defines an interface for all classes that allow to configure a method for quantizing statistics that uses a
     * stochastic rounding strategy.
     */
    class StochasticQuantizationConfig final : public IQuantizationConfig,
                                               public IStochasticQuantizationConfig {
        private:

            uint32 numBits_;

        public:

            StochasticQuantizationConfig();

            uint32 getNumBits() const override;

            IStochasticQuantizationConfig& setNumBits(uint32 numBits) override;

            std::unique_ptr<IQuantizationFactory> createQuantizationFactory() const override;
    };

}
