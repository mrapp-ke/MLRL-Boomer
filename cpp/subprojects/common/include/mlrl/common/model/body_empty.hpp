/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/model/body.hpp"

/**
 * An empty body that does not contain any conditions and therefore covers any examples.
 */
class MLRLCOMMON_API EmptyBody final : public IBody {
    public:

        bool covers(View<const float32>::const_iterator begin, View<const float32>::const_iterator end) const override;

        bool covers(CsrConstView<const float32>::index_const_iterator indicesBegin,
                    CsrConstView<const float32>::index_const_iterator indicesEnd,
                    CsrConstView<const float32>::value_const_iterator valuesBegin,
                    CsrConstView<const float32>::value_const_iterator valuesEnd, View<float32>::iterator tmpArray1,
                    View<uint32>::iterator tmpArray2, uint32 n) const override;

        void visit(EmptyBodyVisitor emptyBodyVisitor, ConjunctiveBodyVisitor conjunctiveBodyVisitor) const override;
};
