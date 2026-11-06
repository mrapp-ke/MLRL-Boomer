#include "mlrl/seco/heuristics/heuristic_wra.hpp"

#include "heuristic_common.hpp"

namespace seco {

    /**
     * An implementation of the type `IHeuristic` that corresponds to the "Weighted Relative Accuracy" (WRA) metric.
     */
    class Wra final : public IHeuristic {
        public:

            float32 evaluateConfusionMatrix(float32 tp, float32 fp, float32 fn, float32 tn) const override {
                return wra(tp, fp, fn, tn);
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
