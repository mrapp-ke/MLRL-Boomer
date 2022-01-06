/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * Allows to create instances of the type `IHeuristic` that allow to trade off between the heuristics "Precision"
     * and "WRA", where the "m" parameter allows to control the trade-off between both heuristics. If m = 0, this
     * heuristic is equivalent to "Precision". As m approaches infinity, the isometrics of this heuristic become
     * equivalent to those of "WRA".
     */
    class MEstimateFactory final : virtual public IHeuristicFactory {

        private:

            float64 m_;

        public:

            MEstimateFactory(float64 m);

            std::unique_ptr<IHeuristic> create() const override;

    };

}
