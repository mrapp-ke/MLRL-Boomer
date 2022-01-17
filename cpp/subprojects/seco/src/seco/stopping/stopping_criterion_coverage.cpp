#include "seco/stopping/stopping_criterion_coverage.hpp"
#include "seco/statistics/statistics.hpp"
#include "common/util/validation.hpp"


namespace seco {

    /**
     * An implementation of the type `IStoppingCriterion` that stops the induction of rules as soon as the sum of the
     * weights of the uncovered labels, as provided by an object of type `ICoverageStatistics`, is smaller than or equal
     * to a certain threshold.
     */
    class CoverageStoppingCriterion final : public IStoppingCriterion {

        private:

            float64 threshold_;

        public:

            /**
             * @param threshold The threshold. Must be at least 0
             */
            CoverageStoppingCriterion(float64 threshold)
                : threshold_(threshold) {

            }

            Result test(const IStatistics& statistics, uint32 numRules) override {
                Result result;
                const ICoverageStatistics& coverageStatistics = static_cast<const ICoverageStatistics&>(statistics);

                if (coverageStatistics.getSumOfUncoveredWeights() > threshold_) {
                    result.action = CONTINUE;
                } else {
                    result.action = FORCE_STOP;
                    result.numRules = numRules;
                }

                return result;
            }

    };

    CoverageStoppingCriterionConfig::CoverageStoppingCriterionConfig()
        : threshold_(0) {

    }

    float64 CoverageStoppingCriterionConfig::getThreshold() const {
        return threshold_;
    }

    CoverageStoppingCriterionConfig& CoverageStoppingCriterionConfig::setThreshold(float64 threshold) {
        assertGreaterOrEqual<float64>("threshold", threshold, 0);
        threshold_ = threshold;
        return *this;
    }

    CoverageStoppingCriterionFactory::CoverageStoppingCriterionFactory(float64 threshold)
        : threshold_(threshold) {

    }

    std::unique_ptr<IStoppingCriterion> CoverageStoppingCriterionFactory::create(
            const SinglePartition& partition) const {
        return std::make_unique<CoverageStoppingCriterion>(threshold_);
    }

    std::unique_ptr<IStoppingCriterion> CoverageStoppingCriterionFactory::create(BiPartition& partition) const {
        return std::make_unique<CoverageStoppingCriterion>(threshold_);
    }

}
