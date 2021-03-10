#include "seco/stopping/stopping_criterion_coverage.hpp"
#include "seco/statistics/statistics_coverage.hpp"
#include "common/debugging/debug.hpp"


namespace seco {

    CoverageStoppingCriterion::CoverageStoppingCriterion(float64 threshold)
        : threshold_(threshold) {

    }

    IStoppingCriterion::Result CoverageStoppingCriterion::test(const IStatistics& statistics, uint32 numRules) {
        const ICoverageStatistics& coverageStatistics = static_cast<const ICoverageStatistics&>(statistics);
        // Debugger: print stopping
        Debugger::printStopping(coverageStatistics.getSumOfUncoveredLabels() <= threshold_);
        return coverageStatistics.getSumOfUncoveredLabels() > threshold_ ? CONTINUE : FORCE_STOP;
    }

}
