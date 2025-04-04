#include "mlrl/seco/heuristics/heuristic_precision.hpp"

#include "heuristic_common.hpp"

namespace seco {

    /**
     * An implementation of the type `IHeuristic` that measures the fraction of correctly predicted labels among all
     * labels that are covered by a rule.
     */
    class Precision final : public IHeuristic {
        public:

            float32 evaluateConfusionMatrix(float32 cin, float32 cip, float32 crn, float32 crp, float32 uin,
                                            float32 uip, float32 urn, float32 urp) const override {
                return precision(cin, cip, crn, crp);
            }
    };

    /**
     * Allows to create instances of the type `IHeuristic` that measure the fraction of correctly predicted labels among
     * all labels that are covered by a rule.
     */
    class PrecisionFactory final : public IHeuristicFactory {
        public:

            std::unique_ptr<IHeuristic> create() const override {
                return std::make_unique<Precision>();
            }
    };

    std::unique_ptr<IHeuristicFactory> PrecisionConfig::createHeuristicFactory() const {
        return std::make_unique<PrecisionFactory>();
    }

}
