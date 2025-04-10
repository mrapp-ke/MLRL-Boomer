#include "mlrl/seco/heuristics/heuristic_recall.hpp"

#include "heuristic_common.hpp"

namespace seco {

    /**
     * An implementation of the type `IHeuristic` that measures the fraction of uncovered labels among all labels for
     * which the rule's prediction is (or would be) correct.
     */
    class Recall final : public IHeuristic {
        public:

            float32 evaluateConfusionMatrix(float32 cin, float32 cip, float32 crn, float32 crp, float32 uin,
                                            float32 uip, float32 urn, float32 urp) const override {
                return recall(cin, crp, uin, urp);
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
