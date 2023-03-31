#include "boosting/prediction/probability_function_logistic.hpp"

#include "boosting/math/math.hpp"

namespace boosting {

    float64 LogisticFunction::transformScoreIntoMarginalProbability(uint32 labelIndex, float64 score) const {
        return logisticFunction(score);
    }

    std::unique_ptr<IMarginalProbabilityFunction> LogisticFunctionFactory::create() const {
        return std::make_unique<LogisticFunction>();
    }

}
