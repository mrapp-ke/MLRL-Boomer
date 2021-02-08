#include "stopping_criterion_coverage.h"
#include "../statistics/statistics_coverage.h"
#include <iostream>

using namespace seco;


CoverageStoppingCriterion::CoverageStoppingCriterion(float64 threshold)
    : threshold_(threshold) {

}

bool CoverageStoppingCriterion::shouldContinue(const IStatistics& statistics, uint32 numRules) {
    const ICoverageStatistics& coverageStatistics = static_cast<const ICoverageStatistics&>(statistics);
    // print out if the seco learner should stop
    if(coverageStatistics.getSumOfUncoveredLabels() <= threshold_) {
        std::cout << "should stop\n";
    }
    return coverageStatistics.getSumOfUncoveredLabels() > threshold_;
}
