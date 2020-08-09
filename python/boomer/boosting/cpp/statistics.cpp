#include "statistics.h"

using namespace boosting;


void AbstractGradientStatistics::resetSampledStatistics() {
    // This function is equivalent to the function `resetCoveredStatistics`...
    this->resetCoveredStatistics();
}

void AbstractGradientStatistics::addSampledStatistic(intp statisticIndex, uint32 weight) {
    // This function is equivalent to the function `updateCoveredStatistic`...
    this->updateCoveredStatistic(statisticIndex, weight, false);
}
