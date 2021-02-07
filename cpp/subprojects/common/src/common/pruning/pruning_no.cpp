#include "common/pruning/pruning_no.hpp"


std::unique_ptr<CoverageMask> NoPruning::prune(IThresholdsSubset& thresholdsSubset, ConditionList& conditions,
                                               const AbstractPrediction& head) const {
    return nullptr;
}
