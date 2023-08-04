#include "mlrl/seco/heuristics/heuristic_laplace.hpp"

namespace seco {

    /**
     * An implementation of the type `IHeuristic` that implements a Laplace-corrected variant of the "Precision" metric.
     */
    class Laplace final : public IHeuristic {
        public:

            float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                            float64 uip, float64 urn, float64 urp) const override {
                float64 numCoveredCorrect = cin + crp;
                float64 numCovered = numCoveredCorrect + cip + crn;
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
