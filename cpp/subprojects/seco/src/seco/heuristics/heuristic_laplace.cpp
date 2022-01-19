#include "seco/heuristics/heuristic_laplace.hpp"


namespace seco {

    /**
     * An implementation of the type `IHeuristic` that implements a Laplace-corrected variant of the "Precision" metric.
     */
    class Laplace final : public IHeuristic {

        public:

            float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                            float64 uip, float64 urn, float64 urp) const override {
                float64 numCoveredIncorrect = cip + crn;
                float64 numCovered = numCoveredIncorrect + cin + crp;
                return (numCoveredIncorrect + 1) / (numCovered + 2);
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

    std::unique_ptr<IHeuristicFactory> LaplaceConfig::create() const {
        return std::make_unique<LaplaceFactory>();
    }

}
