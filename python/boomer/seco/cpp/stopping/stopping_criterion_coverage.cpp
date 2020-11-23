#include "stopping_criterion_coverage.h"
#include "../statistics.h"

using namespace seco;


CoverageStoppingCriterion::CoverageStoppingCriterion(float64 threshold)
    : threshold_(threshold) {

}

bool CoverageStoppingCriterion::shouldContinue(const IStatistics& statistics, uint32 numRules) {
    const ICoverageStatistics& coverageStatistics = static_cast<const ICoverageStatistics&>(statistics);
    return coverageStatistics.getSumOfUncoveredLabels() > threshold_;
}
