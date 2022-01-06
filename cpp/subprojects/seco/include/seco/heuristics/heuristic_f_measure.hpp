/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * Allows to create instances of the type `IHeuristic` that calculate as the (weighted) harmonic mean between the
     * heuristics "Precision" and "Recall", where the parameter "beta" allows to trade off between both heuristics. If
     * beta = 1, both heuristics are weighed equally. If beta = 0, this heuristic is equivalent to "Precision". As beta
     * approaches infinity, this heuristic becomes equivalent to "Recall".
     */
    class FMeasureFactory final : virtual public IHeuristicFactory {

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
