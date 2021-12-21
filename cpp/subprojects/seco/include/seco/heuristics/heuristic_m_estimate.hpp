/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * A heuristic that allows to trade off between the heuristics "Precision2 and "WRA". The "m" parameter allows to
     * control the trade-off between both heuristics. If m = 0, this heuristic is equivalent to "Precision". As m
     * approaches infinity, the isometrics of this heuristic become equivalent to those of "WRA".
     */
    class MEstimate final : public IHeuristic {

        private:

            float64 m_;

        public:

            /**
             * @param m The value of the "m" parameter. Must be at least 0
             */
            MEstimate(float64 m);

            float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                            float64 uip, float64 urn, float64 urp) const override;

    };

    /**
     * Allows to create instances of the type `IHeuristic` that allow to trade off between the heuristics "Precision"
     * and "WRA". The "m" parameter allows to control the trade-off between both heuristics. If m = 0, this heuristic is
     * equivalent to "Precision". As m approaches infinity, the isometrics of this heuristic become equivalent to those
     * of "WRA".
     */
    class MEstimateFactory final : public IHeuristicFactory {

        private:

            float64 m_;

        public:

            MEstimateFactory(float64 m);

            std::unique_ptr<IHeuristic> create() const override;

    };

}
