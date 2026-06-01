#include "mlrl/seco/heuristics/heuristic_laplace.hpp"

namespace seco {

    /**
     * An implementation of the type `IHeuristic` that implements a Laplace-corrected variant of the "Precision" metric.
     */
    class Laplace final : public IHeuristic {
        public:

            float32 evaluateConfusionMatrix(float32 tp, float32 fp, float32 fn, float32 tn) const override {
                return (tp + 1) / (tp + fp + 2);
            }
    };

    /**
     * Allows to create instances of the type `IHeuristic` that implement a Laplace-corrected variant of the "Precision"
     * metric.
     */
    class LaplaceFactory final : public IHeuristicFactory {
        public:

            std::unique_ptr<IHeuristic> create() const override {
                return std::make_unique<Laplace>();
            }
    };

    std::unique_ptr<IHeuristicFactory> LaplaceConfig::createHeuristicFactory() const {
        return std::make_unique<LaplaceFactory>();
    }

}
