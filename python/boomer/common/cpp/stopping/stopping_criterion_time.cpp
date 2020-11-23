#include "stopping_criterion_time.h"


TimeStoppingCriterion::TimeStoppingCriterion(uint32 timeLimit)
    : timeLimit_(timeLimit), startTime_(nullptr) {

}

bool TimeStoppingCriterion::shouldContinue(const IStatistics& statistics, uint32 numRules) {
    if (startTime_ == nullptr) {
        time(startTime_);
        return true;
    } else {
        time_t currentTime;
        time(&currentTime);
        return difftime(currentTime, *startTime_) < timeLimit_;
    }

}
