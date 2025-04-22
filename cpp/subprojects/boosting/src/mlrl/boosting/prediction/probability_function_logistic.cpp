#include "mlrl/boosting/prediction/probability_function_logistic.hpp"

#include "mlrl/boosting/util/math.hpp"

namespace boosting {

    /**
     * An implementation of the class `IMarginalProbabilityFunction` that transforms scores that are predicted for
     * individual labels into marginal probabilities via the logistic sigmoid function.
     */
    class LogisticFunction final : public IMarginalProbabilityFunction {
        private:

            const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel_;

        public:

            /**
             * @param marginalProbabilityCalibrationModel A reference to an object of type
             *                                            `IMarginalProbabilityCalibrationModel` that should be used for
             *                                            the calibration of marginal probabilities
             */
            LogisticFunction(const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel)
                : marginalProbabilityCalibrationModel_(marginalProbabilityCalibrationModel) {}

            float64 transformScoreIntoMarginalProbability(uint32 labelIndex, float32 score) const override {
                float64 marginalProbability = util::logisticFunction(score);
                return marginalProbabilityCalibrationModel_.calibrateMarginalProbability(labelIndex,
                                                                                         marginalProbability);
            }

            float64 transformScoreIntoMarginalProbability(uint32 labelIndex, float64 score) const override {
                float64 marginalProbability = util::logisticFunction(score);
                return marginalProbabilityCalibrationModel_.calibrateMarginalProbability(labelIndex,
                                                                                         marginalProbability);
            }
    };

    std::unique_ptr<IMarginalProbabilityFunction> LogisticFunctionFactory::create(
      const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const {
        return std::make_unique<LogisticFunction>(marginalProbabilityCalibrationModel);
    }

}
