/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * Allows to configure a heuristic that calculates as `1 - wra`, where `wra` corresponds to the "Weighted Relative
     * Accuracy" metric.
     */
    class WraConfig final : public IHeuristicConfig {

    };

    /**
     * Allows to create instances of the type `IHeuristic` that calculate as `1 - wra`, where `wra` corresponds to the
     * "Weighted Relative Accuracy" metric.
     */
    class WraFactory final : public IHeuristicFactory {

        public:

            std::unique_ptr<IHeuristic> create() const override;

    };

}
