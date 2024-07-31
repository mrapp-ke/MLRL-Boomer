#include "mlrl/boosting/statistics/quantization_stochastic.hpp"

#include "mlrl/common/util/validation.hpp"

namespace boosting {

    /**
     * An implementation of the type `IQuantization` that uses a stochastic rounding strategy.
     */
    class StochasticQuantization final : public IQuantization {};

    /**
     * Allows to to create instances of the type `IQuantization` that uses a stochastic rounding strategy.
     */
    class StochasticQuantizationFactory final : public IQuantizationFactory {
        private:

            uint32 numBits_;

        public:

            /**
             * @param numBits The number of bits to be used for quantized statistics
             */
            StochasticQuantizationFactory(uint32 numBits) : numBits_(numBits) {}

            std::unique_ptr<IQuantization> create() const override {
                return std::make_unique<StochasticQuantization>();
            }
    };

    StochasticQuantizationConfig::StochasticQuantizationConfig() : numBits_(4) {}

    uint32 StochasticQuantizationConfig::getNumBits() const {
        return numBits_;
    }

    IStochasticQuantizationConfig& StochasticQuantizationConfig::setNumBits(uint32 numBits) {
        util::assertGreater<uint32>("numBits", numBits, 0);
        numBits_ = numBits;
        return *this;
    }

    std::unique_ptr<IQuantizationFactory> StochasticQuantizationConfig::createQuantizationFactory() const {
        return std::make_unique<StochasticQuantizationFactory>(numBits_);
    }

}
