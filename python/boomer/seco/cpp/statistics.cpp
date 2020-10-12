#include "statistics.h"

using namespace seco;


AbstractCoverageStatistics::AbstractCoverageStatistics(uint32 numStatistics, uint32 numLabels,
                                                       float64 sumUncoveredLabels)
    : AbstractStatistics(numStatistics, numLabels), sumUncoveredLabels_(sumUncoveredLabels) {

}
