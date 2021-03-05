#include "common/pruning/pruning_no.hpp"


std::unique_ptr<ICoverageState> NoPruning::prune(IThresholdsSubset& thresholdsSubset, const IPartition& partition,
                                                 ConditionList& conditions, const AbstractPrediction& head) const {
    return nullptr;
}
