#include "mlrl/seco/heuristics/heuristic_accuracy.hpp"

#include "mlrl/common/util/math.hpp"

namespace seco {

    /**
     * An implementation of the type `IHeuristic` that measures the fraction of correctly predicted labels among all
     * labels.
     */
    class Accuracy final : public IHeuristic {
        public:

            float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                            float64 uip, float64 urn, float64 urp) const override {
                float64 numCoveredCorrect = cin + crp;
                float64 numUncoveredCorrect = uin + uip;
                float64 numCorrect = numCoveredCorrect + numUncoveredCorrect;
                float64 numTotal = numCorrect + cip + crn + urn + urp;
                return util::divideOrZero(numCorrect, numTotal);
            }
    };

    /**
     * Allows to create instances of the type `IHeuristic` that measure the fraction of correctly predicted labels among
     * all labels, i.e., in contrast to the "Precision" metric, examples that are not covered by a rule are taken into
     * account as well.
     */
    class AccuracyFactory final : public IHeuristicFactory {
        public:

            std::unique_ptr<IHeuristic> create() const override {
                return std::make_unique<Accuracy>();
            }
    };

    std::unique_ptr<IHeuristicFactory> AccuracyConfig::createHeuristicFactory() const {
        return std::make_unique<AccuracyFactory>();
    }

}
