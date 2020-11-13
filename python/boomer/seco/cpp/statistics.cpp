#include "statistics.h"

using namespace seco;


AbstractCoverageStatistics::AbstractCoverageStatistics(uint32 numStatistics, uint32 numLabels,
                                                       float64 sumUncoveredLabels)
    : numStatistics_(numStatistics), numLabels_(numLabels), sumUncoveredLabels_(sumUncoveredLabels) {

}

uint32 AbstractCoverageStatistics::getNumStatistics() const {
    return numStatistics_;
}

uint32 AbstractCoverageStatistics::getNumLabels() const {
    return numLabels_;
}
