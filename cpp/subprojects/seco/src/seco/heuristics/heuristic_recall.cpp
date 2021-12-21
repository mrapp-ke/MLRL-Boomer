#include "seco/heuristics/heuristic_recall.hpp"
#include "heuristic_common.hpp"


namespace seco {

    /**
     * An implementation of the type `IHeuristic` that measures the fraction of uncovered labels among all labels for
     * which the rule's prediction is (or would be) correct.
     */
    class Recall final : public IHeuristic {

        public:

            float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                            float64 uip, float64 urn, float64 urp) const override {
                return recall(cin, crp, uin, urp);
            }

    };

    std::unique_ptr<IHeuristic> RecallFactory::create() const {
        return std::make_unique<Recall>();
    }

}
