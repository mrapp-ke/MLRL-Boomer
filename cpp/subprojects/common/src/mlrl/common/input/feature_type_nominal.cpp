#include "mlrl/common/input/feature_type_nominal.hpp"

bool NominalFeatureType::isOrdinal() const {
    return false;
}

bool NominalFeatureType::isNominal() const {
    return true;
}
