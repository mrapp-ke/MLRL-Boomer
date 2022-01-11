/*
 * @author Andreas Seidl Fernandez (aseidlfernandez@gmail.com)
 */
#pragma once

#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * Allows to configure a heuristic that implements a Laplace-corrected variant of the "Precision" metric.
     */
    class LaplaceConfig final : public IHeuristicConfig {

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
