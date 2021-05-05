/**
 * @author Andreas Seidl Fernandez (aseidlfernandez@gmail.com)
 */

#pragma once

#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * A heuristic that is used to prune rules like in IREP: (p + (N - n)) / (P + N), with P(N) all positive
     * (negative) examples and p(n) all positive(negative) covered examples.
     * In Foundations of Rule Learning (Fürnkranz) chapter 7.3.2: This heuristic is called classification accuracy
     */
    class IREP final : public IHeuristic {

    public:

        float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                        float64 uip, float64 urn, float64 urp) const override;

        std::string getName() const override;

    };

}
