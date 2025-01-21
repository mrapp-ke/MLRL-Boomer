#include "mlrl/boosting/rule_evaluation/regularization_no.hpp"

namespace boosting {

    float32 NoRegularizationConfig::getWeight() const {
        return 0.0f;
    }

}
