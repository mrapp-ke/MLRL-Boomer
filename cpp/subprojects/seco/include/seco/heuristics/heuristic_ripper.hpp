/**
 * @author Andreas Seidl Fernandez (aseidlfernandez@gmail.com)
 */

#pragma once

#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * A heuristic that is used to prune rules like in RIPPER: (p - n) / (p + n), with  p(n) all positive(negative)
     * covered examples.
     * As this function is maximised in RIPPER and we try to minimise we used the inverse of the function.
     */
    class RIPPER final : public IHeuristic {

    public:

        float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                        float64 uip, float64 urn, float64 urp) const override;

    };

}