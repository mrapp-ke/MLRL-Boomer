/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/array.hpp"
#include "mlrl/common/data/view_vector_indexed.hpp"
#include "mlrl/common/model/head.hpp"

/**
 * A head that contains a numerical score for a subset of the available labels.
 */
class MLRLCOMMON_API PartialHead final
    : public WritableIndexedVectorDecorator<AllocatedVector<uint32>, AllocatedVector<float64>>,
      public IHead {
    public:

        /**
         * @param numElements The number of scores that are contained by the head
         */
        PartialHead(uint32 numElements);

        void visit(CompleteHeadVisitor completeHeadVisitor, PartialHeadVisitor partialHeadVisitor) const override;
};
