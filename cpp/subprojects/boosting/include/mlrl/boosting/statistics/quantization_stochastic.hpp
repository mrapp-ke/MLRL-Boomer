/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/statistics/quantization.hpp"
#include "mlrl/common/simd/simd.hpp"
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
             * Returns the number of bins that are used for quantized statistics.
             *
             * @return The number of bins that are used
             */
            virtual uint8 getNumBins() const = 0;

            /**
             * Sets the number of bins to be used for quantized statistics.
             *
             * @param numBins   The number of bins to be used. Must be greater than 0
             * @return          A reference to an object of type `IStochasticQuantizationConfig` that allows further
             *                  configuration of the quantization method
             */
            virtual IStochasticQuantizationConfig& setNumBins(uint8 numBins) = 0;
    };

    /**
     * Defines an interface for all classes that allow to configure a method for quantizing statistics that uses a
     * stochastic rounding strategy.
     */
    class StochasticQuantizationConfig final : public IQuantizationConfig,
                                               public IStochasticQuantizationConfig {
        private:

            const ReadableProperty<ISimdConfig> simdConfig_;

            uint8 numBins_;

        public:

            /**
             * @param simdConfig A `ReadableProperty` that allows to access the `ISimdConfig` that stores the
             *                   configuration of SIMD operations
             */
            StochasticQuantizationConfig(ReadableProperty<ISimdConfig> simdConfig);

            uint8 getNumBins() const override;

            IStochasticQuantizationConfig& setNumBins(uint8 numBins) override;

            std::unique_ptr<IQuantizationFactory> createQuantizationFactory(
              const IOutputMatrix& outputMatrix) const override;
    };

}
