/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/stopping/stopping_criterion.hpp"


/**
 * Allows to create instances of the type `IStoppingCriterion` that ensure that a certain time limit is not exceeded.
 */
class TimeStoppingCriterionFactory final : virtual public IStoppingCriterionFactory {

    private:

        uint32 timeLimit_;

    public:

        /**
         * @param timeLimit The time limit in seconds. Must be at least 1
         */
        TimeStoppingCriterionFactory(uint32 timeLimit);

        std::unique_ptr<IStoppingCriterion> create() const override;

};
