/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/seco/heuristics/heuristic.hpp"

#include <memory>

namespace seco {

    /**
     * Allows to configure a heuristic that corresponds to the "Weighted Relative Accuracy" (WRA) metric.
     */
    class WraConfig final : public IHeuristicConfig {
        public:

            std::unique_ptr<IHeuristicFactory> createHeuristicFactory() const override;
    };

}
