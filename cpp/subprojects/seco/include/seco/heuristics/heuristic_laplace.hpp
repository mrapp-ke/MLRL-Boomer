/**
 * @author Andreas Seidl Fernandez (aseidlfernandez@gmail.com)
 */

#pragma once

#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * A heuristic similar to precision but avoids perfect scores.
     */
    class Laplace final : public IHeuristic {

    public:

        float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                        float64 uip, float64 urn, float64 urp) const override;

    };

}