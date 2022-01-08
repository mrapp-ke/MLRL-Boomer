/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/stopping/stopping_criterion.hpp"


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

        std::unique_ptr<IStoppingCriterion> create(const IPartition& partition) const override;

};
