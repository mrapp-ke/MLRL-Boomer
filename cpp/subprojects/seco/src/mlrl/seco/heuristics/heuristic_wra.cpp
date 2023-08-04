#include "mlrl/seco/heuristics/heuristic_wra.hpp"

#include "heuristic_common.hpp"

namespace seco {

    /**
     * An implementation of the type `IHeuristic` that corresponds to the "Weighted Relative Accuracy" (WRA) metric.
     */
    class Wra final : public IHeuristic {
        public:

            float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                            float64 uip, float64 urn, float64 urp) const override {
                return wra(cin, cip, crn, crp, uin, uip, urn, urp);
            }
    };

    /**
     * Allows to create instances of the type `IHeuristic` that corresponds to the "Weighted Relative Accuracy" (WRA)
     * metric.
     */
    class WraFactory final : public IHeuristicFactory {
        public:

            std::unique_ptr<IHeuristic> create() const override {
                return std::make_unique<Wra>();
            }
    };

    std::unique_ptr<IHeuristicFactory> WraConfig::createHeuristicFactory() const {
        return std::make_unique<WraFactory>();
    }

}
