/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * A class that allows to configure a heuristic that calculates as the (weighted) harmonic mean between the
     * heuristics "Precision" and "Recall", where the parameter "beta" allows to trade off between both heuristics. If
     * beta = 1, both heuristics are weighed equally. If beta = 0, this heuristic is equivalent to "Precision". As beta
     * approaches infinity, this heuristic becomes equivalent to "Recall".
     */
    class FMeasureConfig final : public IHeuristicConfig {

        private:

            float64 beta_;

        public:

            FMeasureConfig();

            /**
             * Returns the value of the "beta" parameter.
             *
             * @return The value of the "beta" parameter
             */
            float64 getBeta() const;

            /**
             * Sets the value of the "beta" parameter
             *
             * @param beta  The value of the "beta" parameter. Must be at least 0
             * @return      A reference to an object of type `FMeasureConfig` that allows further configuration of the
             *              heuristic
             */
            FMeasureConfig& setBeta(float64 beta);

    };

    /**
     * Allows to create instances of the type `IHeuristic` that calculate as the (weighted) harmonic mean between the
     * heuristics "Precision" and "Recall", where the parameter "beta" allows to trade off between both heuristics. If
     * beta = 1, both heuristics are weighed equally. If beta = 0, this heuristic is equivalent to "Precision". As beta
     * approaches infinity, this heuristic becomes equivalent to "Recall".
     */
    class FMeasureFactory final : public IHeuristicFactory {

        private:

            float64 beta_;

        public:

            /**
             * @param beta The value of the "beta" parameter. Must be at least 0
             */
            FMeasureFactory(float64 beta);

            std::unique_ptr<IHeuristic> create() const override;

    };

}
