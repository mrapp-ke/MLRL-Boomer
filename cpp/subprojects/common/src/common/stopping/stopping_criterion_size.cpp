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

/**
 * Allows to create instances of the type `IStoppingCriterion` that ensure that the number of induced rules does not
 * exceed a certain maximum.
 */
class SizeStoppingCriterionFactory final : public IStoppingCriterionFactory {

    private:

        const ISizeStoppingCriterionConfig& config_;

    public:

        /**
         * @param config A reference to an object of type `ISizeStoppingCriterionConfig` that specifies the
         *               configuration to be used
         */
        SizeStoppingCriterionFactory(const ISizeStoppingCriterionConfig& config)
            : config_(config) {

        }

        std::unique_ptr<IStoppingCriterion> create(const SinglePartition& partition) const override {
            return std::make_unique<SizeStoppingCriterion>(config_.getMaxRules());
        }

        std::unique_ptr<IStoppingCriterion> create(BiPartition& partition) const override {
            return std::make_unique<SizeStoppingCriterion>(config_.getMaxRules());
        }

};

SizeStoppingCriterionConfig::SizeStoppingCriterionConfig()
    : maxRules_(10) {

}

uint32 SizeStoppingCriterionConfig::getMaxRules() const {
    return maxRules_;
}

ISizeStoppingCriterionConfig& SizeStoppingCriterionConfig::setMaxRules(uint32 maxRules) {
    assertGreaterOrEqual<uint32>("maxRules", maxRules, 1);
    maxRules_ = maxRules;
    return *this;
}

std::unique_ptr<IStoppingCriterionFactory> SizeStoppingCriterionConfig::create() const {
    return std::make_unique<SizeStoppingCriterionFactory>(*this);
}
