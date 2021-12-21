#include "common/pruning/pruning_no.hpp"


std::unique_ptr<ICoverageState> NoPruning::prune(IThresholdsSubset& thresholdsSubset, IPartition& partition,
                                                 ConditionList& conditions, const AbstractPrediction& head) const {
    return nullptr;
}

std::unique_ptr<IPruning> NoPruningFactory::create() const {
    return std::make_unique<NoPruning>();
}
