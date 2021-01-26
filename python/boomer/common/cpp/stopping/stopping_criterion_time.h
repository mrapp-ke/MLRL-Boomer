/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "stopping_criterion.h"
#include <chrono>


/**
 * A stopping criterion that ensures that a certain time limit is not exceeded.
 */
class TimeStoppingCriterion final : public IStoppingCriterion {

    private:

        typedef std::chrono::steady_clock timer;

        typedef std::chrono::seconds timer_unit;

        timer_unit timeLimit_;

        std::chrono::time_point<timer> startTime_;

        bool timerStarted_;

    public:

        /**
         * @param timeLimit The time limit in seconds
         */
        TimeStoppingCriterion(uint32 timeLimit);

        bool shouldContinue(const IStatistics& statistics, uint32 numRules) override;

};
