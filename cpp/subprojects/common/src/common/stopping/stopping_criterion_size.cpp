#include "common/stopping/stopping_criterion_size.hpp"
#include "common/util/validation.hpp"


/**
 * An implementation of the type `IStoppingCriterion` that ensures that the number of induced rules does not exceed a
 * certain maximum.
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

        Result test(const IStatistics& statistics, uint32 numRules) override {
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

SizeStoppingCriterionConfig::SizeStoppingCriterionConfig()
    : maxRules_(1000) {

}

uint32 SizeStoppingCriterionConfig::getMaxRules() const {
    return maxRules_;
}

SizeStoppingCriterionConfig& SizeStoppingCriterionConfig::setMaxRules(uint32 maxRules) {
    assertGreaterOrEqual<uint32>("maxRules", maxRules, 1);
    maxRules_ = maxRules;
    return *this;
}

SizeStoppingCriterionFactory::SizeStoppingCriterionFactory(uint32 maxRules)
    : maxRules_(maxRules) {

}

std::unique_ptr<IStoppingCriterion> SizeStoppingCriterionFactory::create(const SinglePartition& partition) const {
    return std::make_unique<SizeStoppingCriterion>(maxRules_);
}

std::unique_ptr<IStoppingCriterion> SizeStoppingCriterionFactory::create(BiPartition& partition) const {
    return std::make_unique<SizeStoppingCriterion>(maxRules_);
}
