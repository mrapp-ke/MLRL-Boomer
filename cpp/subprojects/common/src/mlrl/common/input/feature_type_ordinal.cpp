#include "mlrl/common/input/feature_type_ordinal.hpp"

bool OrdinalFeatureType::isOrdinal() const {
    return true;
}

bool OrdinalFeatureType::isNominal() const {
    return false;
}
