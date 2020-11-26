/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "statistics_subset.h"


/**
 * An abstract base class for all classes that provide access to a subset of the statistics that are stores by an
 * instance of the class `IHistogram` or `IStatistics` and allow to calculate the scores to be predicted by rules that
 * cover such a subset in the decomposable case, i.e., if the label-wise predictions are the same as the example-wise
 * predictions.
 */
class AbstractDecomposableStatisticsSubset : public IStatisticsSubset {

    public:

        const EvaluatedPrediction& calculateExampleWisePrediction(bool uncovered, bool accumulated) override;

};
