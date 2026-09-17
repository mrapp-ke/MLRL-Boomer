/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector_bit.hpp"
#include "mlrl/common/rule_evaluation/score_vector_decorator.hpp"
#include "mlrl/common/util/quality.hpp"

/**
 * An one-dimensional view that provides access to binary scores that may be predicted by a rule and are stored in a
 * pre-allocated array in a space-efficient way, as well as an overall quality score that assesses the overall quality
 * of the rule.
 *
 * @tparam IndexVector The type of the vector that provides access to the indices of the outputs for which the rule may
 *                     predict
 */
template<typename IndexVector>
class MLRLCOMMON_API BitScoreVectorView : public BitView,
                                          public Quality {
    private:

        const IndexVector& outputIndices_;

        const bool sorted_;

    public:

        /**
         * @param array         A pointer to an array of type `uint32` that stores the values, the view should provide
         *                      access to
         * @param outputIndices A reference to an object of template type `IndexVector` that provides access to the
         *                      indices of the outputs for which the rule may predict
         * @param sorted        True, if the indices of the outputs for which the rule may predict are sorted in
         *                      increasing order, false otherwise
         */
        explicit BitScoreVectorView(uint32* array, const IndexVector& outputIndices, bool sorted)
            : BitView(array, outputIndices.getNumElements()), outputIndices_(outputIndices), sorted_(sorted) {}

        /**
         * @param other A const reference to an object of type `BitScoreVectorView` that should be copied
         */
        BitScoreVectorView(const BitScoreVectorView<IndexVector>& other)
            : BitView(other), outputIndices_(other.outputIndices_), sorted_(other.sorted_) {}

        /**
         * @param other A reference to an object of type `BitScoreVectorView` that should be moved
         */
        BitScoreVectorView(BitScoreVectorView<IndexVector>&& other)
            : BitView(other), outputIndices_(other.outputIndices_), sorted_(other.sorted_) {}

        virtual ~BitScoreVectorView() override {}

        /**
         * The type of the vector that provides access to the indices of the outputs for which the rule may predict.
         */
        using index_vector_type = IndexVector;

        /**
         * The type of the predicted scores.
         */
        using score_type = uint8;

        /**
         * An iterator that provides read-only access to the indices.
         */
        using index_const_iterator = IndexVector::const_iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the indices.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const {
            return outputIndices_.cbegin();
        }

        /**
         * Returns an `index_const_iterator` to the end of the indices.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const {
            return outputIndices_.cend();
        }

        /**
         * Returns the number of outputs for which the rule may predict.
         *
         * @return The number of outputs
         */
        uint32 getNumElements() const {
            return outputIndices_.getNumElements();
        }

        /**
         * Returns whether the rule may only predict for a subset of the available outputs, or not.
         *
         * @return True, if the rule may only predict for a subset of the available outputs, false otherwise
         */
        bool isPartial() const {
            return outputIndices_.isPartial();
        }

        /**
         * Returns whether the indices of the outputs for which the rule may predict are sorted in increasing order, or
         * not.
         *
         * @return True, if the indices of the outputs for which the rule may predict are sorted in increasing order,
         *         false otherwise
         */
        bool isSorted() const {
            return sorted_;
        }
};

/**
 * Allocates the memory, a `BitScoreVectorView` provides access to.
 *
 * @tparam View             The type of the view
 * @tparam MemoryAllocator  The type of the memory allocator to be used
 */
template<typename View, typename MemoryAllocator = DefaultMemoryAllocator>
class MLRLCOMMON_API BitScoreVectorAllocator : public View {
    public:

        /**
         * @param outputIndices A reference to an object of template type `View::index_vector_type` that provides access
         *                      to the indices of the outputs for which the rule may predict
         * @param sorted        True, if the indices of the outputs for which the rule may predict are sorted in
         *                      increasing order, false otherwise
         * @param init          True, if all elements in the view should be value-initialized, false otherwise
         */
        explicit BitScoreVectorAllocator(const typename View::index_vector_type& outputIndices, bool sorted,
                                         bool init = false)
            : View(MemoryAllocator::template allocateMemory<typename View::value_type>(outputIndices.getNumElements(),
                                                                                       init),
                   outputIndices, sorted) {}

        /**
         * @param other A reference to an object of type `DenseScoreVectorAllocator` that should be copied
         */
        BitScoreVectorAllocator(const BitScoreVectorAllocator<View, MemoryAllocator>& other) : View(other) {
            throw std::runtime_error("Objects of type BitScoreVectorAllocator cannot be copied");
        }

        /**
         * @param other A reference to an object of type `BitScoreVectorAllocator` that should be moved
         */
        BitScoreVectorAllocator(BitScoreVectorAllocator<View, MemoryAllocator>&& other) : View(std::move(other)) {
            other.release();
        }

        virtual ~BitScoreVectorAllocator() override {
            MemoryAllocator::freeMemory(View::array);
        }
};

/**
 * An one-dimensional vector that stores binary scores that may be predicted by a rule, as well as an overall quality
 * score that assesses the overall quality of the rule, in a space efficient way.
 *
 * @tparam IndexVector The type of the vector that provides access to the indices of the outputs for which the rule may
 *                     predict
 */
template<typename IndexVector>
class BitScoreVector final
    : public IndexableBitVectorDecorator<
        AbstractScoreVectorViewDecorator<BitScoreVectorAllocator<BitScoreVectorView<IndexVector>>>> {
    public:

        /**
         * @param outputIndices A reference to an object of template type `IndexVector` that provides access to the
         *                      indices of the outputs for which the rule may predict
         * @param sorted        True, if the indices of the outputs for which the rule may predict are sorted in
         *                      increasing order, false otherwise
         */
        BitScoreVector(const IndexVector& outputIndices, bool sorted);

        /**
         * An iterator that provides read-only access to the indices.
         */
        using index_const_iterator = IndexVector::const_iterator;

        /**
         * An iterator that provides read-only access to the predicted scores.
         */
        using value_const_iterator = BitView::const_iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the indices.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the indices.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const;

        /**
         * Returns a `value_const_iterator` to the beginning of the predicted scores.
         *
         * @return A `value_const_iterator` to the beginning
         */
        value_const_iterator values_cbegin() const;

        /**
         * Returns a `value_const_iterator` to the end of the predicted scores.
         *
         * @return A `value_const_iterator` to the end
         */
        value_const_iterator values_cend() const;

        void visit(
          IScoreVector::BitVisitor<CompleteIndexVector> completeBitVisitor,
          IScoreVector::BitVisitor<PartialIndexVector> partialBitVisitor,
          IScoreVector::DenseVisitor<float32, CompleteIndexVector> completeDense32BitVisitor,
          IScoreVector::DenseVisitor<float32, PartialIndexVector> partialDense32BitVisitor,
          IScoreVector::DenseVisitor<float64, CompleteIndexVector> completeDense64BitVisitor,
          IScoreVector::DenseVisitor<float64, PartialIndexVector> partialDense64BitVisitor,
          IScoreVector::DenseBinnedVisitor<float32, CompleteIndexVector> completeDenseBinned32BitVisitor,
          IScoreVector::DenseBinnedVisitor<float32, PartialIndexVector> partialDenseBinned32BitVisitor,
          IScoreVector::DenseBinnedVisitor<float64, CompleteIndexVector> completeDenseBinned64BitVisitor,
          IScoreVector::DenseBinnedVisitor<float64, PartialIndexVector> partialDenseBinned64BitVisitor) const override;
};
