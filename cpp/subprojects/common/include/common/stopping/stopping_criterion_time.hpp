/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/stopping/stopping_criterion.hpp"


/**
 * Allows to configure a stopping criterion that ensures that a certain time limit is not exceeded.
 */
class TimeStoppingCriterionConfig final : public IStoppingCriterionConfig {

    private:

        uint32 timeLimit_;

    public:

        TimeStoppingCriterionConfig();

        /**
         * Returns the time limit.
         *
         * @return The time limit in seconds
         */
        uint32 getTimeLimit() const;

        /**
         * Sets the time limit.
         *
         * @param timeLimit The time limit in seconds. Must be at least 1
         * @return          A reference to an object of type `TimeStoppingCriterionConfig` that allows further
         *                  configuration of the stopping criterion
         */
        TimeStoppingCriterionConfig& setTimeLimit(uint32 timeLimit);

};

/**
 * Allows to create instances of the type `IStoppingCriterion` that ensure that a certain time limit is not exceeded.
 */
class TimeStoppingCriterionFactory final : public IStoppingCriterionFactory {

    private:

        uint32 timeLimit_;

    public:

        /**
         * @param timeLimit The time limit in seconds. Must be at least 1
         */
        TimeStoppingCriterionFactory(uint32 timeLimit);

        std::unique_ptr<IStoppingCriterion> create(const SinglePartition& partition) const override;

        std::unique_ptr<IStoppingCriterion> create(BiPartition& partition) const override;

};
