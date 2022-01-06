#include "seco/heuristics/heuristic_accuracy.hpp"


namespace seco {

    /**
     * An implementation of the type `IHeuristic` that measures the fraction of incorrectly predicted labels among all
     * labels.
     */
    class Accuracy final : virtual public IHeuristic {

        public:

            float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                            float64 uip, float64 urn, float64 urp) const override {
                float64 numUncoveredCorrect = urp + uin;
                float64 numCoveredIncorrect = cip + crn;
                float64 numTotal = numUncoveredCorrect + numCoveredIncorrect + cin + crp + uip + urn;
                return (numUncoveredCorrect + numCoveredIncorrect) / numTotal;
            }

    };

    std::unique_ptr<IHeuristic> AccuracyFactory::create() const {
        return std::make_unique<Accuracy>();
    }

}
