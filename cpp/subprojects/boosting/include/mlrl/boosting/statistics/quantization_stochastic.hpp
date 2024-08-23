/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/statistics/quantization.hpp"
#include "mlrl/common/random/rng.hpp"
#include "mlrl/common/util/properties.hpp"

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

            const ReadableProperty<RNGConfig> rngConfig_;

            uint32 numBits_;

        public:

            /**
             * @param rngConfig A `ReadableProperty` that provides access the `RNGConfig` that stores the configuration
             *                  of random number generators
             */
            StochasticQuantizationConfig(ReadableProperty<RNGConfig> rngConfig);

            uint32 getNumBits() const override;

            IStochasticQuantizationConfig& setNumBits(uint32 numBits) override;

            std::unique_ptr<IQuantizationFactory> createQuantizationFactory() const override;
    };

}
