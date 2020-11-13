#include "statistics.h"

using namespace boosting;


AbstractGradientStatistics::AbstractGradientStatistics(uint32 numStatistics, uint32 numLabels)
    : numStatistics_(numStatistics), numLabels_(numLabels) {

}

uint32 AbstractGradientStatistics::getNumStatistics() const {
    return numStatistics_;
}

uint32 AbstractGradientStatistics::getNumLabels() const {
    return numLabels_;
}

void AbstractGradientStatistics::resetSampledStatistics() {
    // This function is equivalent to the function `resetCoveredStatistics`...
    this->resetCoveredStatistics();
}

void AbstractGradientStatistics::addSampledStatistic(uint32 statisticIndex, uint32 weight) {
    // This function is equivalent to the function `updateCoveredStatistic`...
    this->updateCoveredStatistic(statisticIndex, weight, false);
}
