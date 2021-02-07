#include "stopping_criterion_time.hpp"


TimeStoppingCriterion::TimeStoppingCriterion(uint32 timeLimit)
    : timeLimit_(std::chrono::duration_cast<timer_unit>(std::chrono::seconds(timeLimit))), startTime_(timer::now()),
      timerStarted_(false) {

}

bool TimeStoppingCriterion::shouldContinue(const IStatistics& statistics, uint32 numRules) {
    if (timerStarted_) {
        auto currentTime = timer::now();
        auto duration = std::chrono::duration_cast<timer_unit>(currentTime - startTime_);
        return duration < timeLimit_;
    } else {
        startTime_ = timer::now();
        timerStarted_ = true;
        return true;
    }

}
