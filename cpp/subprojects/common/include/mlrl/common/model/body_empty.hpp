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

        bool covers(View<uint32>::const_iterator indicesBegin, View<uint32>::const_iterator indicesEnd,
                    View<float32>::const_iterator valuesBegin, View<float32>::const_iterator valuesEnd,
                    float32 sparseValue, View<float32>::iterator tmpArray1, View<uint32>::iterator tmpArray2,
                    uint32 n) const override;

        void visit(std::optional<EmptyBodyVisitor> emptyBodyVisitor,
                   std::optional<ConjunctiveBodyVisitor> conjunctiveBodyVisitor) const override;
};
