#include "mlrl/seco/heuristics/heuristic_accuracy.hpp"

#include "mlrl/common/math/scalar_math.hpp"

namespace seco {

    /**
     * An implementation of the type `IHeuristic` that measures the fraction of correctly predicted labels among all
     * labels.
     */
    class Accuracy final : public IHeuristic {
        public:

            float32 evaluateConfusionMatrix(float32 tp, float32 fp, float32 fn, float32 tn) const override {
                float32 numCorrect = tp + fn;
                float32 numTotal = numCorrect + fp + tn;
                return math::divideOrZero(numCorrect, numTotal);
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
