#include "mlrl/seco/stopping/stopping_criterion_coverage.hpp"

#include "mlrl/common/util/validation.hpp"
#include "mlrl/seco/statistics/statistics.hpp"

namespace seco {

    /**
     * An implementation of the type `IStoppingCriterion` that stops the induction of rules as soon as the sum of the
     * weights of the uncovered labels, as provided by an object of type `ICoverageStatistics`, is smaller than or equal
     * to a certain threshold.
     */
    class CoverageStoppingCriterion final : public IStoppingCriterion {
        private:

            const float64 threshold_;

        public:

            /**
             * @param threshold The threshold. Must be at least 0
             */
            CoverageStoppingCriterion(float64 threshold) : threshold_(threshold) {}

            Result test(const IStatistics& statistics, uint32 numRules) override {
                Result result;
                const ICoverageStatistics& coverageStatistics = dynamic_cast<const ICoverageStatistics&>(statistics);

                if (!(coverageStatistics.getSumOfUncoveredWeights() > threshold_)) {
                    result.stop = true;
                }

                return result;
            }
    };

    /**
     * Allows to create instances of the type `IStoppingCriterion` that stop the induction of rules as soon as the sum
     * of the weights of the uncovered labels, as provided by an object of type `ICoverageStatistics`, is smaller or
     * equal to a certain threshold.
     */
    class CoverageStoppingCriterionFactory final : public IStoppingCriterionFactory {
        private:

            const float64 threshold_;

        public:

            /**
             * @param threshold The threshold. Must be at least 0
             */
            CoverageStoppingCriterionFactory(float64 threshold) : threshold_(threshold) {}

            std::unique_ptr<IStoppingCriterion> create(const SinglePartition& partition) const override {
                return std::make_unique<CoverageStoppingCriterion>(threshold_);
            }

            std::unique_ptr<IStoppingCriterion> create(BiPartition& partition) const override {
                return std::make_unique<CoverageStoppingCriterion>(threshold_);
            }
    };

    CoverageStoppingCriterionConfig::CoverageStoppingCriterionConfig() : threshold_(0) {}

    float64 CoverageStoppingCriterionConfig::getThreshold() const {
        return threshold_;
    }

    ICoverageStoppingCriterionConfig& CoverageStoppingCriterionConfig::setThreshold(float64 threshold) {
        util::assertGreaterOrEqual<float64>("threshold", threshold, 0);
        threshold_ = threshold;
        return *this;
    }

    std::unique_ptr<IStoppingCriterionFactory> CoverageStoppingCriterionConfig::createStoppingCriterionFactory() const {
        return std::make_unique<CoverageStoppingCriterionFactory>(threshold_);
    }

}
