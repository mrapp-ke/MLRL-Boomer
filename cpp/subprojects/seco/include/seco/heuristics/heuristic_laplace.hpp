/*
 * @author Andreas Seidl Fernandez (aseidlfernandez@gmail.com)
 */
#pragma once

#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * Laplace-corrected variant of the "Precision" metric.
     */
    class Laplace final : public IHeuristic {

        public:

            float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                            float64 uip, float64 urn, float64 urp) const override;

    };

    /**
     * Allows to create instances of the type `IHeuristic` that implement a Laplace-corrected variant of the "Precision"
     * metric.
     */
    class LaplaceFactory final : public IHeuristicFactory {

        public:

            std::unique_ptr<IHeuristic> create() const override;

    };

}
