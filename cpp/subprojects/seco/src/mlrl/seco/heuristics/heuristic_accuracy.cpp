#include "mlrl/seco/heuristics/heuristic_accuracy.hpp"

#include "mlrl/common/util/math.hpp"

namespace seco {

    /**
     * An implementation of the type `IHeuristic` that measures the fraction of correctly predicted labels among all
     * labels.
     */
    class Accuracy final : public IHeuristic {
        public:

            float32 evaluateConfusionMatrix(float32 cin, float32 cip, float32 crn, float32 crp, float32 uin,
                                            float32 uip, float32 urn, float32 urp) const override {
                float32 numCoveredCorrect = cin + crp;
                float32 numUncoveredCorrect = uin + uip;
                float32 numCorrect = numCoveredCorrect + numUncoveredCorrect;
                float32 numTotal = numCorrect + cip + crn + urn + urp;
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
