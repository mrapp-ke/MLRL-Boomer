#include "common/stopping/stopping_criterion_size.hpp"
#include "common/validation.hpp"


/**
 * A stopping criterion that ensures that the number of induced rules does not exceed a certain maximum.
 */
class SizeStoppingCriterion final : public IStoppingCriterion {

    private:

        uint32 maxRules_;

    public:

        /**
         * @param maxRules The maximum number of rules. Must be at least 1
         */
        SizeStoppingCriterion(uint32 maxRules)
            : maxRules_(maxRules) {

        }

        Result test(const IPartition& partition, const IStatistics& statistics, uint32 numRules) override {
            Result result;

            if (numRules < maxRules_) {
                result.action = CONTINUE;
            } else {
                result.action = FORCE_STOP;
                result.numRules = numRules;
            }

            return result;
        }

};

SizeStoppingCriterionFactory::SizeStoppingCriterionFactory(uint32 maxRules)
    : maxRules_(maxRules) {
    assertGreaterOrEqual<uint32>("maxRules", maxRules, 1);
}

std::unique_ptr<IStoppingCriterion> SizeStoppingCriterionFactory::create() const {
    return std::make_unique<SizeStoppingCriterion>(maxRules_);
}
