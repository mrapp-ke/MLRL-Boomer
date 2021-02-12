#include "common/stopping/stopping_criterion_size.hpp"


SizeStoppingCriterion::SizeStoppingCriterion(uint32 maxRules)
    : maxRules_(maxRules) {

}

IStoppingCriterion::Action SizeStoppingCriterion::test(const IStatistics& statistics, uint32 numRules) {
    return numRules < maxRules_ ? CONTINUE : FORCE_STOP;
}
