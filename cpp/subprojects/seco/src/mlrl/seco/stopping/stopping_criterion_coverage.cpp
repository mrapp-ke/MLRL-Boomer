#include "mlrl/seco/stopping/stopping_criterion_coverage.hpp"

#include "mlrl/common/util/validation.hpp"
#include "mlrl/seco/statistics/statistics.hpp"

namespace seco {

    /**
     * An implementation of the type `IStoppingCriterion` that stops the induction of rules as soon as a certain
     * fraction of the available training examples and labels is covered.
     */
    class CoverageStoppingCriterion final : public IStoppingCriterion {
        private:

            const float32 minCoverage_;

        public:

            /**
             * @param minCoverage The fraction of training examples and labels that must be covered before the induction
             *                    of rules is stopped. Must be in [0, 1)
             */
            CoverageStoppingCriterion(float32 minCoverage) : minCoverage_(minCoverage) {}

            Result test(const IStatistics& statistics, uint32 numRules) override {
                Result result;
                const ICoverageStatistics& coverageStatistics = dynamic_cast<const ICoverageStatistics&>(statistics);

                if (!(coverageStatistics.getUncoveredFraction() > minCoverage_)) {
                    result.stop = true;
                }

                return result;
            }
    };

    /**
     * Allows to create instances of the type `IStoppingCriterion` that stop the induction of rules as soon as a certain
     * fraction of the training examples and labels is covered.
     */
    class CoverageStoppingCriterionFactory final : public IStoppingCriterionFactory {
        private:

            const float32 minCoverage_;

        public:

            /**
             * @param minCoverage The fraction of training examples and labels that must be covered before the induction
             *                    of rules is stopped. Must be in [0, 1)
             */
            CoverageStoppingCriterionFactory(float32 minCoverage) : minCoverage_(minCoverage) {}

            std::unique_ptr<IStoppingCriterion> create(const SinglePartition& partition) const override {
                return std::make_unique<CoverageStoppingCriterion>(minCoverage_);
            }

            std::unique_ptr<IStoppingCriterion> create(BiPartition& partition) const override {
                return std::make_unique<CoverageStoppingCriterion>(minCoverage_);
            }
    };

    CoverageStoppingCriterionConfig::CoverageStoppingCriterionConfig() : minCoverage_(0.0f) {}

    float32 CoverageStoppingCriterionConfig::getMinCoverage() const {
        return minCoverage_;
    }

    ICoverageStoppingCriterionConfig& CoverageStoppingCriterionConfig::setMinCoverage(float32 minCoverage) {
        util::assertGreaterOrEqual<float32>("minCoverage", minCoverage, 0);
        util::assertLess<float32>("minCoverage", minCoverage, 1);
        minCoverage_ = minCoverage;
        return *this;
    }

    std::unique_ptr<IStoppingCriterionFactory> CoverageStoppingCriterionConfig::createStoppingCriterionFactory() const {
        return std::make_unique<CoverageStoppingCriterionFactory>(minCoverage_);
    }

}
