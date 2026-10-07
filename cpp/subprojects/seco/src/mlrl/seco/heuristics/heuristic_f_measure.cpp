#include "mlrl/seco/heuristics/heuristic_f_measure.hpp"

#include "heuristic_common.hpp"
#include "mlrl/common/util/validation.hpp"

namespace seco {

    /**
     * An implementation of the type `IHeuristic` that calculates as the (weighted) harmonic mean between the heuristics
     * "Precision" and "Recall", where the parameter "beta" allows to trade off between both heuristics.
     */
    class FMeasure final : public IHeuristic {
        private:

            const float32 beta_;

        public:

            /**
             * @param beta The value of the "beta" parameter. Must be at least 0
             */
            FMeasure(float32 beta) : beta_(beta) {}

            float32 evaluateConfusionMatrix(float32 tp, float32 fp, float32 fn, float32 tn) const override {
                if (std::isinf(beta_)) {
                    // Equivalent to recall
                    return recall(tp, fn);
                } else if (beta_ > 0) {
                    // Weighted harmonic mean between precision and recall
                    float32 betaPow = beta_ * beta_;
                    float32 numerator = (1 + betaPow) * tp;
                    float32 denominator = numerator + (betaPow * fn) + fp;
                    return math::divideOrZero(numerator, denominator);
                } else {
                    // Equivalent to precision
                    return precision(tp, fp);
                }
            }
    };

    /**
     * Allows to create instances of the type `IHeuristic` that calculate as the (weighted) harmonic mean between the
     * heuristics "Precision" and "Recall", where the parameter "beta" allows to trade off between both heuristics. If
     * beta = 1, both heuristics are weighed equally. If beta = 0, this heuristic is equivalent to "Precision". As beta
     * approaches infinity, this heuristic becomes equivalent to "Recall".
     */
    class FMeasureFactory final : public IHeuristicFactory {
        private:

            const float32 beta_;

        public:

            /**
             * @param beta The value of the "beta" parameter. Must be at least 0
             */
            FMeasureFactory(float32 beta) : beta_(beta) {}

            std::unique_ptr<IHeuristic> create() const override {
                return std::make_unique<FMeasure>(beta_);
            }
    };

    FMeasureConfig::FMeasureConfig() : beta_(0.25f) {}

    float32 FMeasureConfig::getBeta() const {
        return beta_;
    }

    IFMeasureConfig& FMeasureConfig::setBeta(float32 beta) {
        util::assertGreaterOrEqual<float32>("beta", beta, 0);
        beta_ = beta;
        return *this;
    }

    std::unique_ptr<IHeuristicFactory> FMeasureConfig::createHeuristicFactory() const {
        return std::make_unique<FMeasureFactory>(beta_);
    }

}
