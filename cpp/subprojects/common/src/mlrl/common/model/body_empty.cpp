#include "mlrl/common/model/body_empty.hpp"

bool EmptyBody::covers(View<const float32>::const_iterator begin, View<const float32>::const_iterator end) const {
    return true;
}

bool EmptyBody::covers(View<uint32>::const_iterator indicesBegin, View<uint32>::const_iterator indicesEnd,
                       View<float32>::const_iterator valuesBegin, View<float32>::const_iterator valuesEnd,
                       float32 sparseValue, float32* tmpArray1, uint32* tmpArray2, uint32 n) const {
    return true;
}

void EmptyBody::visit(std::optional<EmptyBodyVisitor> emptyBodyVisitor,
                      std::optional<ConjunctiveBodyVisitor> conjunctiveBodyVisitor) const {
    if (emptyBodyVisitor) {
        (*emptyBodyVisitor)(*this);
    }
}
