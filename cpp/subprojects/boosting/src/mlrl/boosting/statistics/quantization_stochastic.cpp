#include "mlrl/boosting/statistics/quantization_stochastic.hpp"

#include "mlrl/common/math/vector_math.hpp"
#include "mlrl/common/simd/vector_math.hpp"
#include "mlrl/common/util/validation.hpp"

namespace boosting {

    /**
     * An implementation of the type `IQuantization` that uses a stochastic rounding strategy.
     */
    class StochasticQuantization final : public IQuantization {};

    /**
     * Allows to to create instances of the type `IQuantization` that uses a stochastic rounding strategy.
     */
    template<typename VectorMath>
    class StochasticQuantizationFactory final : public IQuantizationFactory {
        private:

            uint8 numBins_;

        public:

            /**
             * @param numBins The number of bins to be used for quantized statistics
             */
            StochasticQuantizationFactory(uint8 numBins) : numBins_(numBins) {}

            std::unique_ptr<IQuantization> create() const override {
                return std::make_unique<StochasticQuantization>();
            }
    };

    StochasticQuantizationConfig::StochasticQuantizationConfig(ReadableProperty<ISimdConfig> simdConfig)
        : simdConfig_(simdConfig), numBins_(16) {}

    uint8 StochasticQuantizationConfig::getNumBins() const {
        return numBins_;
    }

    IStochasticQuantizationConfig& StochasticQuantizationConfig::setNumBins(uint8 numBins) {
        util::assertGreater<uint8>("numBins", numBins, 0);
        numBins_ = numBins;
        return *this;
    }

    std::unique_ptr<IQuantizationFactory> StochasticQuantizationConfig::createQuantizationFactory(
      const IOutputMatrix& outputMatrix) const {
#if SIMD_SUPPORT_ENABLED
        if (simdConfig_.get().isSimdRecommended(outputMatrix.getNumOutputs())) {
            return std::make_unique<StochasticQuantizationFactory<SimdVectorMath>>(numBins_);
        }
#endif

        return std::make_unique<StochasticQuantizationFactory<SequentialVectorMath>>(numBins_);
    }

}
