/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "pruning.hpp"


/**
 * An implementation of the class `IPruning` that does not actually perform any pruning, but retains all conditions.
 */
class NoPruning final : public IPruning {

    public:

        std::unique_ptr<CoverageMask> prune(IThresholdsSubset& thresholdsSubset, ConditionList& conditions,
                                            const AbstractPrediction& head) const override;

};
