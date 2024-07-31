#include "mlrl/boosting/statistics/quantization_no.hpp"

namespace boosting {

    /**
     * An implementation of the type `IQuantization` that does not actually perform any quantization.
     */
    class NoQuantization final : public IQuantization {};

    /**
     * Allows to to create instances of the type `IQuantization` that do not actually perform any quantization.
     */
    class NoQuantizationFactory final : public IQuantizationFactory {
        public:

            std::unique_ptr<IQuantization> create() const override {
                return std::make_unique<NoQuantization>();
            }
    };

    std::unique_ptr<IQuantizationFactory> NoQuantizationConfig::createQuantizationFactory() const {
        return std::make_unique<NoQuantizationFactory>();
    }

}
