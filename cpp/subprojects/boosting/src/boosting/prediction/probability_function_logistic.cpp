#include "boosting/prediction/probability_function_logistic.hpp"

#include "boosting/math/math.hpp"

namespace boosting {

    /**
     * An implementation of the class `IMarginalProbabilityFunction` that transforms regression scores that are
     * predicted for individual labels into marginal probabilities via the logistic sigmoid function.
     */
    class LogisticFunction final : public IMarginalProbabilityFunction {
        private:

            const IProbabilityCalibrationModel& probabilityCalibrationModel_;

        public:

            /**
             * @param probabilityCalibrationModel A reference to an object of type `IProbabilityCalibrationModel` that
             *                                    should be used for the calibration of probabilities
             */
            LogisticFunction(const IProbabilityCalibrationModel& probabilityCalibrationModel)
                : probabilityCalibrationModel_(probabilityCalibrationModel) {}

            float64 transformScoreIntoMarginalProbability(uint32 labelIndex, float64 score) const override {
                return probabilityCalibrationModel_.calibrateMarginalProbability(labelIndex, logisticFunction(score));
            }
    };

    std::unique_ptr<IMarginalProbabilityFunction> LogisticFunctionFactory::create(
      const IProbabilityCalibrationModel& probabilityCalibrationModel) const {
        return std::make_unique<LogisticFunction>(probabilityCalibrationModel);
    }

}
