#include "mlrl/seco/heuristics/heuristic_wra.hpp"

#include "heuristic_common.hpp"

namespace seco {

    /**
     * An implementation of the type `IHeuristic` that corresponds to the "Weighted Relative Accuracy" (WRA) metric.
     */
    class Wra final : public IHeuristic {
        public:

            float32 evaluateConfusionMatrix(float32 cin, float32 cip, float32 crn, float32 crp, float32 uin,
                                            float32 uip, float32 urn, float32 urp) const override {
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
