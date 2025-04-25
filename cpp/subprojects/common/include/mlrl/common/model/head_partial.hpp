/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector_indexed.hpp"
#include "mlrl/common/model/head.hpp"

#include <type_traits>

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
        PartialHead(uint32 numElements)
            : IterableIndexedVectorDecorator<
                IndexedVectorDecorator<AllocatedVector<uint32>, AllocatedVector<ScoreType>>>(
                CompositeVector<AllocatedVector<uint32>, AllocatedVector<ScoreType>>(
                  AllocatedVector<uint32>(numElements), AllocatedVector<ScoreType>(numElements))) {}

        void visit(CompleteHeadVisitor<uint8> completeBinaryHeadVisitor,
                   CompleteHeadVisitor<float32> complete32BitHeadVisitor,
                   CompleteHeadVisitor<float64> complete64BitHeadVisitor,
                   PartialHeadVisitor<uint8> partialBinaryHeadVisitor,
                   PartialHeadVisitor<float32> partial32BitHeadVisitor,
                   PartialHeadVisitor<float64> partial64BitHeadVisitor) const override {
            if constexpr (std::is_same_v<ScoreType, uint8>) {
                partialBinaryHeadVisitor(*this);
            } else if constexpr (std::is_same_v<ScoreType, float32>) {
                partial32BitHeadVisitor(*this);
            } else if constexpr (std::is_same_v<ScoreType, float64>) {
                partial64BitHeadVisitor(*this);
            } else {
                throw std::runtime_error("No visitor available for handling object of template class PartialHead");
            }
        }
};
