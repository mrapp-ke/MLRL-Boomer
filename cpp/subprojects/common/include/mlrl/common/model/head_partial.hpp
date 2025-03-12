/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector_indexed.hpp"
#include "mlrl/common/model/head.hpp"

/**
 * A head that contains a numerical score for a subset of the available outputs.
 *
 * @tparam ScoreType The type of the numerical scores
 */
template<typename ScoreType>
class MLRLCOMMON_API PartialHead final : public IterableIndexedVectorDecorator<
                                           IndexedVectorDecorator<AllocatedVector<uint32>, AllocatedVector<ScoreType>>>,
                                         public IHead {
    public:

        /**
         * @param numElements The number of scores that are contained by the head
         */
        PartialHead(uint32 numElements);

        void visit(CompleteHeadVisitor<float32> complete32BitHeadVisitor,
                   CompleteHeadVisitor<float64> complete64BitHeadVisitor,
                   PartialHeadVisitor<float32> partial32BitHeadVisitor,
                   PartialHeadVisitor<float64> partial64BitHeadVisitor) const override;
};
