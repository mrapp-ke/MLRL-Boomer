#include "seco/heuristics/heuristic_f_measure.hpp"
#include "common/util/validation.hpp"
#include "heuristic_common.hpp"
#include <cmath>


namespace seco {

    /**
     * An implementation of the type `IHeuristic` that calculates as the (weighted) harmonic mean between the heuristics
     * "Precision" and "Recall", where the parameter "beta" allows to trade off between both heuristics.
     */
    class FMeasure final : public IHeuristic {

        private:

            float64 beta_;

        public:

            /**
             * @param beta The value of the "beta" parameter. Must be at least 0
             */
            FMeasure(float64 beta)
                : beta_(beta) {

            }

            float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                            float64 uip, float64 urn, float64 urp) const override {
                if (std::isinf(beta_)) {
                    // Equivalent to recall
                    return recall(cin, crp, uin, urp);
                } else if (beta_ > 0) {
                    // Weighted harmonic mean between precision and recall
                    float64 numCoveredEqual = cin + crp;
                    float64 betaPow = beta_ * beta_;
                    float64 numerator = (1 + betaPow) * numCoveredEqual;
                    float64 denominator = numerator + (betaPow * (uin + urp)) + (cip + crn);

                    if (denominator == 0) {
                        return 1;
                    }

                    return 1 - (numerator / denominator);
                } else {
                    // Equivalent to precision
                    return precision(cin, cip, crn, crp);
                }
            }

    };

    FMeasureConfig::FMeasureConfig()
        : beta_(0.25) {

    }

    float64 FMeasureConfig::getBeta() const {
        return beta_;
    }

    FMeasureConfig& FMeasureConfig::setBeta(float64 beta) {
        beta_ = beta;
        return *this;
    }

    FMeasureFactory::FMeasureFactory(float64 beta)
        : beta_(beta) {
        assertGreaterOrEqual<float64>("beta", beta, 0);
    }

    std::unique_ptr<IHeuristic> FMeasureFactory::create() const {
        return std::make_unique<FMeasure>(beta_);
    }

}
