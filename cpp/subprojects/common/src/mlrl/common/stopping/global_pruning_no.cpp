#include "mlrl/common/stopping/global_pruning_no.hpp"

std::unique_ptr<IStoppingCriterionFactory> NoGlobalPruningConfig::createStoppingCriterionFactory() const {
    return nullptr;
}

bool NoGlobalPruningConfig::shouldUseHoldoutSet() const {
    return false;
}

bool NoGlobalPruningConfig::shouldRemoveUnusedRules() const {
    return false;
}
