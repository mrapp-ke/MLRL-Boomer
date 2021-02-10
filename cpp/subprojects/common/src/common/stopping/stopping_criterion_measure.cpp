#include "common/stopping/stopping_criterion_measure.hpp"


MeasureStoppingCriterion::MeasureStoppingCriterion(std::shared_ptr<IMeasure> measurePtr)
    : measurePtr_(measurePtr) {

}

bool MeasureStoppingCriterion::shouldContinue(const IPartition& partition, const IStatistics& statistics,
                                              uint32 numRules) {
    // TODO Implement
    return true;
}
