#include "seco/stopping/stopping_criterion_coverage.hpp"
#include "seco/statistics/statistics_coverage.hpp"


namespace seco {

    CoverageStoppingCriterion::CoverageStoppingCriterion(float64 threshold)
        : threshold_(threshold) {

    }

    IStoppingCriterion::Result CoverageStoppingCriterion::test(const IStatistics& statistics, uint32 numRules) {
        Result result;
        const ICoverageStatistics& coverageStatistics = static_cast<const ICoverageStatistics&>(statistics);
        result.action = coverageStatistics.getSumOfUncoveredLabels() > threshold_ ? CONTINUE : FORCE_STOP;
        return result;
    }

}
