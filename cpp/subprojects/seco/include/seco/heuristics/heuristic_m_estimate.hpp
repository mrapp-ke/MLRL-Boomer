/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "seco/heuristics/heuristic.hpp"
#include "seco/macros.hpp"

namespace seco {

    /**
     * Defines an interface for all classes that allow to configure a heuristic that trades off between the heuristics
     * "Precision" and "WRA", where the "m" parameter controls the trade-off between both heuristics. If m = 0, this
     * heuristic is equivalent to "Precision". As m approaches infinity, the isometrics of this heuristic become
     * equivalent to those of "WRA".
     */
    class MLRLSECO_API IMEstimateConfig {
        public:

            virtual ~IMEstimateConfig() {};

            /**
             * Returns the value of the "m" parameter.
             *
             * @return The value of the "m" parameter
             */
            virtual float64 getM() const = 0;

            /**
             * Sets the value of the "m" parameter.
             *
             * @param m The value of the "m" parameter. Must be at least 0
             * @return  A reference to an object of type `IMEstimateConfig` that allows further configuration of the
             *          heuristic
             */
            virtual IMEstimateConfig& setM(float64 m) = 0;
    };

    /**
     * Allows to configure a heuristic that trades off between the heuristics "Precision" and "WRA", where the "m"
     * parameter controls the trade-off between both heuristics. If m = 0, this heuristic is equivalent to "Precision".
     * As m approaches infinity, the isometrics of this heuristic become equivalent to those of "WRA".
     */
    class MEstimateConfig final : public IHeuristicConfig, public IMEstimateConfig {
        private:

            float64 m_;

        public:

            MEstimateConfig();

            float64 getM() const override;

            IMEstimateConfig& setM(float64 m) override;

            std::unique_ptr<IHeuristicFactory> createHeuristicFactory() const override;
    };

}
