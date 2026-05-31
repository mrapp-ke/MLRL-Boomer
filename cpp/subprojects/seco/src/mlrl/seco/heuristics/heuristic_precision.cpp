#include "mlrl/seco/heuristics/heuristic_precision.hpp"

#include "heuristic_common.hpp"

namespace seco {

    /**
     * An implementation of the type `IHeuristic` that measures the fraction of correctly predicted labels among all
     * labels that are covered by a rule.
     */
    class Precision final : public IHeuristic {
        public:

            float32 evaluateConfusionMatrix(float32 tp, float32 fp, float32 fn, float32 tn) const override {
                return precision(tp, fp);
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
