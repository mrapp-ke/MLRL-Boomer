#include "common/stopping/stopping_criterion_time.hpp"


TimeStoppingCriterion::TimeStoppingCriterion(uint32 timeLimit)
    : timeLimit_(std::chrono::duration_cast<timer_unit>(std::chrono::seconds(timeLimit))), startTime_(timer::now()),
      timerStarted_(false) {

}

IStoppingCriterion::Result TimeStoppingCriterion::test(const IStatistics& statistics, uint32 numRules) {
    Result result;

    if (timerStarted_) {
        auto currentTime = timer::now();
        auto duration = std::chrono::duration_cast<timer_unit>(currentTime - startTime_);
        result.action = duration < timeLimit_ ? CONTINUE : FORCE_STOP;
    } else {
        startTime_ = timer::now();
        timerStarted_ = true;
        result.action = CONTINUE;
    }

    return result;
}
