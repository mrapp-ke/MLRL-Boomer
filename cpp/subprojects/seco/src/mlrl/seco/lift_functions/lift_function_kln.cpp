#include "mlrl/seco/lift_functions/lift_function_kln.hpp"

#include "mlrl/common/util/validation.hpp"

namespace seco {

    static inline float32 calculateLiftInternally(uint32 numLabels, float32 k) {
        return 1.0 + (k * std::log((float32) numLabels));
    }

    /**
     * A lift function that monotonously increases according to the natural algorithm of the number of labels for which
     * a rule predicts.
     */
    class KlnLiftFunction final : public ILiftFunction {
        private:

            const float32 k_;

            const float32 maxLift_;

        public:

            /**
             * @param k         The value of the parameter "k". The steepness of the lift function increases with larger
             *                  values for "k". Must be at least 0
             * @param maxLift   The maximum possible lift
             */
            KlnLiftFunction(float32 k, float32 maxLift) : k_(k), maxLift_(maxLift) {}

            float32 calculateLift(uint32 numLabels) const override {
                return calculateLiftInternally(numLabels, k_);
            }

            float32 getMaxLift(uint32 numLabels) const override {
                return maxLift_;
            }
    };

    /**
     * Allows to create instances of the type `ILiftFunction` that monotonously increase according to the natural
     * logarithm of the number of labels for which a rule predicts.
     */
    class KlnLiftFunctionFactory final : public ILiftFunctionFactory {
        private:

            const float32 k_;

            const float32 maxLift_;

        public:

            /**
             * @param numLabels The total number of available labels. Must be greater than 0
             * @param k         The value of the parameter "k". The steepness of the lift function increases with larger
             *                  values for "k". Must be at least 0
             */
            KlnLiftFunctionFactory(uint32 numLabels, float32 k)
                : k_(k), maxLift_(calculateLiftInternally(numLabels, k)) {}

            std::unique_ptr<ILiftFunction> create() const override {
                return std::make_unique<KlnLiftFunction>(k_, maxLift_);
            }
    };

    KlnLiftFunctionConfig::KlnLiftFunctionConfig() : k_(0.2f) {}

    float32 KlnLiftFunctionConfig::getK() const {
        return k_;
    }

    IKlnLiftFunctionConfig& KlnLiftFunctionConfig::setK(float32 k) {
        util::assertGreater<float32>("k", k, 0);
        k_ = k;
        return *this;
    }

    std::unique_ptr<ILiftFunctionFactory> KlnLiftFunctionConfig::createLiftFunctionFactory(
      const IRowWiseLabelMatrix& labelMatrix) const {
        uint32 numLabels = labelMatrix.getNumOutputs();
        return std::make_unique<KlnLiftFunctionFactory>(numLabels, k_);
    }

}
