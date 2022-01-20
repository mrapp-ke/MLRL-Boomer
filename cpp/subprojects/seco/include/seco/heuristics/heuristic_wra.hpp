/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * Defines an interface for all classes that allow to configure a heuristic that calculates as `1 - wra`, where
     * `wra` corresponds to the "Weighted Relative Accuracy" metric.
     */
    class IWraConfig {

        public:

            virtual ~IWraConfig() { };

    };

    /**
     * Allows to configure a heuristic that calculates as `1 - wra`, where `wra` corresponds to the "Weighted Relative
     * Accuracy" metric.
     */
    class WraConfig final : public IHeuristicConfig, public IWraConfig {

        public:

            std::unique_ptr<IHeuristicFactory> configure() const override;

    };

}
