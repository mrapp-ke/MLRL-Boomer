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

        public:

            std::unique_ptr<IHeuristicFactory> createHeuristicFactory() const override;

    };

}
