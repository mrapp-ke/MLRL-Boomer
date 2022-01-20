/*
 * @author Andreas Seidl Fernandez (aseidlfernandez@gmail.com)
 */
#pragma once

#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * Defines an interface for all classes that allow to configure a heuristic that implements a Laplace-corrected
     * variant of the "Precision" metric.
     */
    class ILaplaceConfig {

        public:

            virtual ~ILaplaceConfig() { };

    };

    /**
     * Allows to configure a heuristic that implements a Laplace-corrected variant of the "Precision" metric.
     */
    class LaplaceConfig final : public IHeuristicConfig, public ILaplaceConfig {

        public:

            std::unique_ptr<IHeuristicFactory> configure() const override;

    };

}
