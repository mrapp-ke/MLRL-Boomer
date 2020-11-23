/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "stopping_criterion.h"
#include <ctime>


/**
 * A stopping criterion that ensures that a certain time limit is not exceeded.
 */
class TimeStoppingCriterion : public IStoppingCriterion {

    private:

        uint32 timeLimit_;

        time_t startTime_;

    public:

        /**
         * @param timeLimit The time limit in seconds
         */
        TimeStoppingCriterion(uint32 timeLimit);

        bool shouldContinue(const IStatistics& statistics, uint32 numRules) override;

};
