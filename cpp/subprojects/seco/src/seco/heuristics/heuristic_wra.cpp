#include "seco/heuristics/heuristic_wra.hpp"
#include "heuristic_common.hpp"


namespace seco {

    /**
     * An implementation of the type `IHeuristic` that calculates as `1 - wra`, where `wra` corresponds to the "Weighted
     * Relative Accuracy" metric.
     */
    class Wra final : virtual public IHeuristic {

        public:

            float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                            float64 uip, float64 urn, float64 urp) const override {
                return wra(cin, cip, crn, crp, uin, uip, urn, urp);
            }

    };

    std::unique_ptr<IHeuristic> WraFactory::create() const {
        return std::make_unique<Wra>();
    }

}
