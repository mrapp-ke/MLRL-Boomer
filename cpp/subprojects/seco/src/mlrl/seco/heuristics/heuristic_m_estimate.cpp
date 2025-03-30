#include "mlrl/seco/heuristics/heuristic_m_estimate.hpp"

#include "heuristic_common.hpp"
#include "mlrl/common/util/validation.hpp"

namespace seco {

    /**
     * An implementation of the type `IHeuristic` that allows to trade off between the heuristics "Precision" and "WRA",
     * where the "m" parameter allows to control the trade-off between both heuristics.
     */
    class MEstimate final : public IHeuristic {
        private:

            const float32 m_;

        public:

            /**
             * @param m The value of the "m" parameter. Must be at least 0
             */
            MEstimate(float32 m) : m_(m) {}

            float32 evaluateConfusionMatrix(float32 cin, float32 cip, float32 crn, float32 crp, float32 uin,
                                            float32 uip, float32 urn, float32 urp) const override {
                if (std::isinf(m_)) {
                    // Equivalent to weighted relative accuracy
                    return wra(cin, cip, crn, crp, uin, uip, urn, urp);
                } else if (m_ > 0) {
                    // Trade-off between precision and weighted relative accuracy
                    float32 numCoveredEqual = cin + crp;
                    float32 numCovered = numCoveredEqual + cip + crn;
                    float32 numUncoveredEqual = uin + urp;
                    float32 numEqual = numCoveredEqual + numUncoveredEqual;
                    float32 numTotal = numCovered + numUncoveredEqual + uip + urn;

                    if (numTotal > 0) {
                        return (numCoveredEqual + (m_ * (numEqual / numTotal))) / (numCovered + m_);
                    }

                    return 0;
                } else {
                    // Equivalent to precision
                    return precision(cin, cip, crn, crp);
                }
            }
    };

    /**
     * Allows to create instances of the type `IHeuristic` that trade off between the heuristics "Precision" and "WRA",
     * where the "m" parameter controls the trade-off between both heuristics. If m = 0, this heuristic is equivalent to
     * "Precision". As m approaches infinity, the isometrics of this heuristic become equivalent to those of "WRA".
     */
    class MEstimateFactory final : public IHeuristicFactory {
        private:

            const float32 m_;

        public:

            /**
             * @param The value of the "m" parameter. Must be at least 0
             */
            MEstimateFactory(float32 m) : m_(m) {}

            std::unique_ptr<IHeuristic> create() const override {
                return std::make_unique<MEstimate>(m_);
            }
    };

    MEstimateConfig::MEstimateConfig() : m_(22.466f) {}

    float32 MEstimateConfig::getM() const {
        return m_;
    }

    IMEstimateConfig& MEstimateConfig::setM(float32 m) {
        util::assertGreaterOrEqual<float32>("m", m, 0);
        m_ = m;
        return *this;
    }

    std::unique_ptr<IHeuristicFactory> MEstimateConfig::createHeuristicFactory() const {
        return std::make_unique<MEstimateFactory>(m_);
    }

}
