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
        public:

            Result test(const IStatistics& statistics, uint32 numRules) override {
                Result result;
                const ICoverageStatistics& coverageStatistics = dynamic_cast<const ICoverageStatistics&>(statistics);

                if (!(coverageStatistics.getSumOfUncoveredWeights() > 0)) {
                    result.stop = true;
                }

                return result;
            }
    };

    /**
     * Allows to create instances of the type `IStoppingCriterion` that stop the induction of rules as soon as the
     * entire label space is covered.
     */
    class CoverageStoppingCriterionFactory final : public IStoppingCriterionFactory {
        public:

            std::unique_ptr<IStoppingCriterion> create(const SinglePartition& partition) const override {
                return std::make_unique<CoverageStoppingCriterion>();
            }

            std::unique_ptr<IStoppingCriterion> create(BiPartition& partition) const override {
                return std::make_unique<CoverageStoppingCriterion>();
            }
    };

    std::unique_ptr<IStoppingCriterionFactory> CoverageStoppingCriterionConfig::createStoppingCriterionFactory() const {
        return std::make_unique<CoverageStoppingCriterionFactory>();
    }

}
