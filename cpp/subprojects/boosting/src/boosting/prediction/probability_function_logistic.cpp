#include "boosting/prediction/probability_function_logistic.hpp"

#include "boosting/math/math.hpp"

namespace boosting {

    /**
     * An implementation of the class `IMarginalProbabilityFunction` that transforms regression scores that are
     * predicted for individual labels into marginal probabilities via the logistic sigmoid function.
     */
    class LogisticFunction final : public IMarginalProbabilityFunction {
        public:

            float64 transformScoreIntoMarginalProbability(float64 score) const override {
                return logisticFunction(score);
            }
    };

    std::unique_ptr<IMarginalProbabilityFunction> LogisticFunctionFactory::create() const {
        return std::make_unique<LogisticFunction>();
    }

}
