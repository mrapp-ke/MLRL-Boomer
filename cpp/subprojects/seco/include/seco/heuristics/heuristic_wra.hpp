/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * A heuristic that calculates as `1 - wra`, where `wra` corresponds to the weighted relative accuracy metric.
     */
    class WRA final : public IHeuristic {

        public:

            float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                            float64 uip, float64 urn, float64 urp) const override;

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
