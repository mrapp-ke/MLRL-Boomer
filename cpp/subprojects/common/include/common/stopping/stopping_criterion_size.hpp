/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/stopping/stopping_criterion.hpp"


/**
 * Allows to configure a stopping criterion that ensures that the number of induced rules does not exceed a certain
 * maximum.
 */
class SizeStoppingCriterionConfig final : public IStoppingCriterionConfig {

    private:

        uint32 maxRules_;

    public:

        SizeStoppingCriterionConfig();

        /**
         * Returns the maximum number of rules that are induced.
         *
         * @return The maximum number of rules that are induced
         */
        uint32 getMaxRules() const;

        /**
         * Sets the maximum number of rules that should be induced.
         *
         * @param maxRules  The maximum number of rules that should be induced. Must be at least 1
         * @return          A reference to an object of type `SizeStoppingCriterionConfig` that allows further
         *                  configuration of the stopping criterion
         */
        SizeStoppingCriterionConfig& setMaxRules(uint32 maxRules);

};

/**
 * Allows to create instances of the type `IStoppingCriterion` that ensure that the number of induced rules does not
 * exceed a certain maximum.
 */
class SizeStoppingCriterionFactory final : public IStoppingCriterionFactory {

    private:

        uint32 maxRules_;

    public:

        /**
         * @param maxRules The maximum number of rules. Must be at least 1
         */
        SizeStoppingCriterionFactory(uint32 maxRules);

        std::unique_ptr<IStoppingCriterion> create(const SinglePartition& partition) const override;

        std::unique_ptr<IStoppingCriterion> create(BiPartition& partition) const override;

};
