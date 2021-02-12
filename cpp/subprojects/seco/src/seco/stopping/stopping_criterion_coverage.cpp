#include "seco/stopping/stopping_criterion_coverage.hpp"
#include "seco/statistics/statistics_coverage.hpp"


namespace seco {

    CoverageStoppingCriterion::CoverageStoppingCriterion(float64 threshold)
        : threshold_(threshold) {

    }

    IStoppingCriterion::Action CoverageStoppingCriterion::test(const IStatistics& statistics, uint32 numRules) {
        const ICoverageStatistics& coverageStatistics = static_cast<const ICoverageStatistics&>(statistics);
        return coverageStatistics.getSumOfUncoveredLabels() > threshold_ ? CONTINUE : FORCE_STOP;
    }

}
