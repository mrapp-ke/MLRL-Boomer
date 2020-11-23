#include "stopping_criterion_size.h"


SizeStoppingCriterion::SizeStoppingCriterion(uint32 maxRules)
    : maxRules_(maxRules) {

}

bool SizeStoppingCriterion::shouldContinue(const IStatistics& statistics, uint32 numRules) {
    return numRules < maxRules_;
}
