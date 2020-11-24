/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "pruning.h"


/**
 * Implements incremental reduced error pruning (IREP) for pruning classification rules.
 *
 * Given `n` conditions in the order of their induction, IREP allows to remove up to `n - 1` trailing conditions,
 * depending on which of the resulting rules improves the most over the quality score of the original rules as measured
 * on the prune set.
 */
class IREP : virtual public IPruning {

    public:

        virtual std::unique_ptr<CoverageMask> prune(IThresholdsSubset& thresholdsSubset, ConditionList& conditions,
                                                    const AbstractPrediction& head) const override;

};
