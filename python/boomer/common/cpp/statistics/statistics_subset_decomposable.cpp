#include "statistics_subset_decomposable.h"


const DenseScoreVector& AbstractDecomposableStatisticsSubset::calculateExampleWiseScores(bool uncovered,
                                                                                         bool accumulated) {
    // In the decomposable case, the example-wise predictions are the same as the label-wise predictions...
    return this->calculateLabelWiseScores(uncovered, accumulated);
}
