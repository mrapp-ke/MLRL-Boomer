#include "mlrl/seco/heuristics/heuristic_laplace.hpp"

namespace seco {

    /**
     * An implementation of the type `IHeuristic` that implements a Laplace-corrected variant of the "Precision" metric.
     */
    class Laplace final : public IHeuristic {
        public:

            float32 evaluateConfusionMatrix(float32 cin, float32 cip, float32 crn, float32 crp, float32 uin,
                                            float32 uip, float32 urn, float32 urp) const override {
                float32 numCoveredCorrect = cin + crp;
                float32 numCovered = numCoveredCorrect + cip + crn;
                return (numCoveredCorrect + 1) / (numCovered + 2);
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
