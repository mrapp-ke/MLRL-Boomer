#include "mlrl/boosting/rule_evaluation/regularization_manual.hpp"

#include "mlrl/common/util/validation.hpp"

namespace boosting {

    ManualRegularizationConfig::ManualRegularizationConfig() : regularizationWeight_(1.0f) {}

    float32 ManualRegularizationConfig::getRegularizationWeight() const {
        return regularizationWeight_;
    }

    IManualRegularizationConfig& ManualRegularizationConfig::setRegularizationWeight(float32 regularizationWeight) {
        util::assertGreater<float32>("regularizationWeight", regularizationWeight, 0);
        regularizationWeight_ = regularizationWeight;
        return *this;
    }

    float32 ManualRegularizationConfig::getWeight() const {
        return regularizationWeight_;
    }

}
