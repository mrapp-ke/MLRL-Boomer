#include "mlrl/common/model/body_empty.hpp"

bool EmptyBody::covers(View<const float32>::const_iterator begin, View<const float32>::const_iterator end) const {
    return true;
}

bool EmptyBody::covers(CsrView<const float32>::index_const_iterator indicesBegin,
                       CsrView<const float32>::index_const_iterator indicesEnd,
                       CsrView<const float32>::value_const_iterator valuesBegin,
                       CsrView<const float32>::value_const_iterator valuesEnd, float32* tmpArray1, uint32* tmpArray2,
                       uint32 n) const {
    return true;
}

void EmptyBody::visit(EmptyBodyVisitor emptyBodyVisitor, ConjunctiveBodyVisitor conjunctiveBodyVisitor) const {
    emptyBodyVisitor(*this);
}
