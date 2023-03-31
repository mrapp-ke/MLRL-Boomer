#include "boosting/prediction/probability_function_logistic.hpp"

#include "boosting/math/math.hpp"

namespace boosting {

    LogisticFunction::LogisticFunction(const IProbabilityCalibrationModel& probabilityCalibrationModel)
        : probabilityCalibrationModel_(probabilityCalibrationModel) {}

    float64 LogisticFunction::transformScoreIntoMarginalProbability(uint32 labelIndex, float64 score) const {
        return probabilityCalibrationModel_.calibrateMarginalProbability(labelIndex, logisticFunction(score));
    }

    std::unique_ptr<IMarginalProbabilityFunction> LogisticFunctionFactory::create(
      const IProbabilityCalibrationModel& probabilityCalibrationModel) const {
        return std::make_unique<LogisticFunction>(probabilityCalibrationModel);
    }

}
