#include "stopping_criterion_time.h"


TimeStoppingCriterion::TimeStoppingCriterion(uint32 timeLimit)
    : timeLimit_(timeLimit), startTime_(0) {

}

bool TimeStoppingCriterion::shouldContinue(const IStatistics& statistics, uint32 numRules) {
    time_t currentTime;

    if (startTime_ == 0) {
        startTime_ = time(currentTime);
        return true;
    }

    return difftime(currentTime, startTime_) < timeLimit_;
}
