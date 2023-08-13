#include "mlrl/common/input/feature_type_numerical.hpp"

bool NumericalFeatureType::isOrdinal() const {
    return false;
}

bool NumericalFeatureType::isNominal() const {
    return false;
}
