#include "common/stopping/stopping_criterion_time.hpp"
#include "common/validation.hpp"
#include <chrono>


/**
 * An implementation of the type `IStoppingCriterion` that ensures that a certain time limit is not exceeded.
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
         * @param timeLimit The time limit in seconds. Must be at least 1
         */
        TimeStoppingCriterion(uint32 timeLimit)
            : timeLimit_(std::chrono::duration_cast<timer_unit>(std::chrono::seconds(timeLimit))),
              startTime_(timer::now()), timerStarted_(false) {

        }

        Result test(const IPartition& partition, const IStatistics& statistics, uint32 numRules) override {
            Result result;

            if (timerStarted_) {
                auto currentTime = timer::now();
                auto duration = std::chrono::duration_cast<timer_unit>(currentTime - startTime_);

                if (duration < timeLimit_) {
                    result.action = CONTINUE;
                } else {
                    result.action = FORCE_STOP;
                    result.numRules = numRules;
                }
            } else {
                startTime_ = timer::now();
                timerStarted_ = true;
                result.action = CONTINUE;
            }

            return result;
        }

};

TimeStoppingCriterionFactory::TimeStoppingCriterionFactory(uint32 timeLimit)
    : timeLimit_(timeLimit) {
    assertGreaterOrEqual<uint32>("timeLimit", timeLimit, 1);
}

std::unique_ptr<IStoppingCriterion> TimeStoppingCriterionFactory::create() const {
    return std::make_unique<TimeStoppingCriterion>(timeLimit_);
}
