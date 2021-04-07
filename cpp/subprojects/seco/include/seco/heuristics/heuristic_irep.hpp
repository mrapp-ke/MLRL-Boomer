/**
 * @author Andreas Seidl Fernandez (aseidlfernandez@gmail.com)
 */

#pragma once

#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * A heuristic that is used to prune rules like in IREP: (p + (N - n)) / (P + N), with P(N) all positive
     * (negative) examples and p(n) all positive(negative) covered examples.
     * As this function is maximised in IREP and we try to minimise we used the inverse of the function.
     */
    class IREP final : public IHeuristic {

    public:

        float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                        float64 uip, float64 urn, float64 urp) const override;

    };

}
