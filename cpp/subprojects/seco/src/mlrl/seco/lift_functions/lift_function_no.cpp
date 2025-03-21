#include "mlrl/seco/lift_functions/lift_function_no.hpp"

namespace seco {

    /**
     * A lift function that does not affect the quality of rules.
     */
    class NoLiftFunction final : public ILiftFunction {
        public:

            float32 calculateLift(uint32 numLabels) const override {
                return 1;
            }

            float32 getMaxLift(uint32 numLabels) const override {
                return 1;
            }
    };

    /**
     * Allows to create instances of the type `ILiftFunction` that does not affect the quality of rules.
     */
    class NoLiftFunctionFactory final : public ILiftFunctionFactory {
        public:

            std::unique_ptr<ILiftFunction> create() const override {
                return std::make_unique<NoLiftFunction>();
            }
    };

    std::unique_ptr<ILiftFunctionFactory> NoLiftFunctionConfig::createLiftFunctionFactory(
      const IRowWiseLabelMatrix& labelMatrix) const {
        return std::make_unique<NoLiftFunctionFactory>();
    }

}
