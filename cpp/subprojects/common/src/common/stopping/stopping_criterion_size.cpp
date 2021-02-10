#include "common/stopping/stopping_criterion_size.hpp"


SizeStoppingCriterion::SizeStoppingCriterion(uint32 maxRules)
    : maxRules_(maxRules) {

}

bool SizeStoppingCriterion::shouldContinue(const IPartition& IPartition, const IStatistics& statistics,
                                           uint32 numRules) {
    return numRules < maxRules_;
}
