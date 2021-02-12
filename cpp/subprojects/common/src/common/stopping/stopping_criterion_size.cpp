#include "common/stopping/stopping_criterion_size.hpp"


SizeStoppingCriterion::SizeStoppingCriterion(uint32 maxRules)
    : maxRules_(maxRules) {

}

IStoppingCriterion::Result SizeStoppingCriterion::test(const IStatistics& statistics, uint32 numRules) {
    Result result;
    result.action = numRules < maxRules_ ? CONTINUE : FORCE_STOP;
    return result;
}
