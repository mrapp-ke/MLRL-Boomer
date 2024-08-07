#include "mlrl/common/stopping/stopping_criterion_no.hpp"

std::unique_ptr<IStoppingCriterionFactory> NoStoppingCriterionConfig::createStoppingCriterionFactory() const {
    return nullptr;
}
