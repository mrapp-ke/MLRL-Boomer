#include "seco/stopping/stopping_criterion_coverage.hpp"
#include "seco/statistics/statistics_coverage.hpp"
#include "common//debugging/global.hpp"
#include <iostream>


namespace seco {

    CoverageStoppingCriterion::CoverageStoppingCriterion(float64 threshold)
        : threshold_(threshold) {

    }

    IStoppingCriterion::Result CoverageStoppingCriterion::test(const IStatistics& statistics, uint32 numRules) {
        const ICoverageStatistics& coverageStatistics = static_cast<const ICoverageStatistics&>(statistics);
        // print out if the seco learner should stop
        if (coverageStatistics.getSumOfUncoveredLabels() <= threshold_ && debugging_ == 1) {
            std::cout << "should stop\n";
        }
        return coverageStatistics.getSumOfUncoveredLabels() > threshold_ ? CONTINUE : FORCE_STOP;
    }

}
