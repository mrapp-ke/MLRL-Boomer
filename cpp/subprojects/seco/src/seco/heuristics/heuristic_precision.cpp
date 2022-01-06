#include "seco/heuristics/heuristic_precision.hpp"
#include "heuristic_common.hpp"


namespace seco {

    /**
     * An implementation of the type `IHeuristic` that measures the fraction of incorrectly predicted labels among all
     * labels that are covered by a rule.
     */
    class Precision final : virtual public IHeuristic {

        public:

            float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                            float64 uip, float64 urn, float64 urp) const override {
                return precision(cin, cip, crn, crp);
            }

    };

    std::unique_ptr<IHeuristic> PrecisionFactory::create() const {
        return std::make_unique<Precision>();
    }

}
