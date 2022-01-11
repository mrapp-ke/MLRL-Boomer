/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * Allows to configure a heuristic that trades off between the heuristics "Precision" and "WRA", where the "m"
     * parameter controls the trade-off between both heuristics. If m = 0, this heuristic is equivalent to "Precision".
     * As m approaches infinity, the isometrics of this heuristic become equivalent to those of "WRA".
     */
    class MEstimateConfig final : public IHeuristicConfig {

        private:

            float64 m_;

        public:

            MEstimateConfig();

            /**
             * Returns the value of the "m" parameter.
             *
             * @return The value of the "m" parameter
             */
            float64 getM() const;

            /**
             * Sets the value of the "m" parameter.
             *
             * @param The value of the "m" parameter. Must be at least 0
             */
            MEstimateConfig& setM(float64 m);

    };

    /**
     * Allows to create instances of the type `IHeuristic` that trade off between the heuristics "Precision" and "WRA",
     * where the "m" parameter controls the trade-off between both heuristics. If m = 0, this heuristic is equivalent to
     * "Precision". As m approaches infinity, the isometrics of this heuristic become equivalent to those of "WRA".
     */
    class MEstimateFactory final : public IHeuristicFactory {

        private:

            float64 m_;

        public:

            /**
             * @param The value of the "m" parameter. Must be at least 0
             */
            MEstimateFactory(float64 m);

            std::unique_ptr<IHeuristic> create() const override;

    };

}
