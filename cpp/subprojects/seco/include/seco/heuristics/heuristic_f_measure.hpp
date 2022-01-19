/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * Defines an interface for all classes that allow to configure a heuristic that calculates as the (weighted)
     * harmonic mean between the heuristics "Precision" and "Recall", where the parameter "beta" allows to trade off
     * between both heuristics. If beta = 1, both heuristics are weighed equally. If beta = 0, this heuristic is
     * equivalent to "Precision". As beta approaches infinity, this heuristic becomes equivalent to "Recall".
     */
    class IFMeasureConfig {

        public:

            virtual ~IFMeasureConfig() { };

            /**
             * Returns the value of the "beta" parameter.
             *
             * @return The value of the "beta" parameter
             */
            virtual float64 getBeta() const = 0;

            /**
             * Sets the value of the "beta" parameter
             *
             * @param beta  The value of the "beta" parameter. Must be at least 0
             * @return      A reference to an object of type `IFMeasureConfig` that allows further configuration of the
             *              heuristic
             */
            virtual IFMeasureConfig& setBeta(float64 beta) = 0;

    };

    /**
     * Allows to configure a heuristic that calculates as the (weighted) harmonic mean between the heuristics
     * "Precision" and "Recall", where the parameter "beta" allows to trade off between both heuristics. If beta = 1,
     * both heuristics are weighed equally. If beta = 0, this heuristic is equivalent to "Precision". As beta approaches
     * infinity, this heuristic becomes equivalent to "Recall".
     */
    class FMeasureConfig final : public IHeuristicConfig, public IFMeasureConfig {

        private:

            float64 beta_;

        public:

            FMeasureConfig();

            float64 getBeta() const override;

            IFMeasureConfig& setBeta(float64 beta) override;

            std::unique_ptr<IHeuristicFactory> create() const override;

    };

}
