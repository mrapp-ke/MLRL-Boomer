#include "pruning_no.h"


std::unique_ptr<CoverageMask> NoPruning::prune(IThresholdsSubset& thresholdsSubset, ConditionList& conditions,
                                               const AbstractPrediction& head) const {
    return nullptr;
}
