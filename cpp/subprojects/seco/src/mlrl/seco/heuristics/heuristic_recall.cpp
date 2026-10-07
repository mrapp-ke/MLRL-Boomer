#include "mlrl/seco/heuristics/heuristic_recall.hpp"

#include "heuristic_common.hpp"

namespace seco {

    /**
     * An implementation of the type `IHeuristic` that measures the fraction of uncovered labels among all labels for
     * which the rule's prediction is (or would be) correct.
     */
    class Recall final : public IHeuristic {
        public:

            float32 evaluateConfusionMatrix(float32 tp, float32 fp, float32 fn, float32 tn) const override {
                return recall(tp, fn);
            }
    };

    /**
     * Allows to create instances of the type `IHeuristic` that measure the fraction of uncovered labels among all
     * labels for which a rule's prediction is (or would be) correct, i.e., for which the ground truth is equal to the
     * rule's prediction.
     */
    class RecallFactory final : public IHeuristicFactory {
        public:

            std::unique_ptr<IHeuristic> create() const override {
                return std::make_unique<Recall>();
            }
    };

    std::unique_ptr<IHeuristicFactory> RecallConfig::createHeuristicFactory() const {
        return std::make_unique<RecallFactory>();
    }

}
