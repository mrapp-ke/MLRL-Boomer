/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/rule_evaluation/score_vector.hpp"
#include "mlrl/common/util/quality.hpp"

/**
 * A one-dimensional view that provides access to scores that may be predicted by a rule and are stored in a
 * pre-allocated array of a specific size, as well as an overall quality score that assesses the quality of the rule.
 *
 * @tparam ScoreType    The type of the predicted scores
 * @tparam IndexVector  The type of the vector that provides access to the indices of the outputs for which the rule may
 *                      predict
 */
template<typename ScoreType, typename IndexVector>
class MLRLCOMMON_API DenseScoreVectorView : public View<ScoreType>,
                                            public Quality {
    private:

        const IndexVector& outputIndices_;

        const bool sorted_;

    public:

        /**
         * @param array         A pointer to an array of template type `ScoreType` that stores the values, the view
         *                      should provide access to
         * @param outputIndices A reference to an object of template type `IndexVector` that provides access to the
         *                      indices of the outputs for which the rule may predict
         * @param sorted        True, if the indices of the outputs for which the rule may predict are sorted in
         *                      increasing order, false otherwise
         */
        explicit DenseScoreVectorView(ScoreType* array, const IndexVector& outputIndices, bool sorted)
            : View<ScoreType>(array, outputIndices.getNumElements()), outputIndices_(outputIndices), sorted_(sorted) {}

        /**
         * @param other A const reference to an object of type `DenseScoreVectorView` that should be copied
         */
        DenseScoreVectorView(const DenseScoreVectorView<ScoreType, IndexVector>& other)
            : View<ScoreType>(other), outputIndices_(other.outputIndices_), sorted_(other.sorted_) {}

        /**
         * @param other A reference to an object of type `DenseScoreVectorView` that should be moved
         */
        DenseScoreVectorView(DenseScoreVectorView<ScoreType, IndexVector>&& other)
            : View<ScoreType>(other), outputIndices_(other.outputIndices_), sorted_(other.sorted_) {}

        virtual ~DenseScoreVectorView() override {}

        /**
         * The type of the vector that provides access to the indices of the outputs for which the rule may predict.
         */
        using index_vector_type = IndexVector;

        /**
         * An iterator that provides read-only access to the indices.
         */
        using index_const_iterator = IndexVector::const_iterator;

        /**
         * Returns a `const_iterator` to the end of the predicted scores.
         *
         * @return A `const_iterator` to the end
         */
        typename View<ScoreType>::const_iterator cend() const {
            return &View<ScoreType>::array[this->getNumElements()];
        }

        /**
         * Returns an `iterator` to the end of the predicted scores.
         *
         * @return An `iterator` to the end
         */
        typename View<ScoreType>::iterator end() {
            return &View<ScoreType>::array[this->getNumElements()];
        }

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
 * Allocates the memory, a `DenseScoreVectorView` provides access to.
 *
 * @tparam View             The type of the view
 * @tparam MemoryAllocator  The type of the memory allocator to be used
 */
template<typename View, typename MemoryAllocator = DefaultMemoryAllocator>
class MLRLCOMMON_API DenseScoreVectorAllocator : public View {
    public:

        /**
         * @param outputIndices A reference to an object of template type `View::index_vector_type` that provides access
         *                      to the indices of the outputs for which the rule may predict
         * @param sorted        True, if the indices of the outputs for which the rule may predict are sorted in
         *                      increasing order, false otherwise
         * @param init          True, if all elements in the view should be value-initialized, false otherwise
         */
        explicit DenseScoreVectorAllocator(const typename View::index_vector_type& outputIndices, bool sorted,
                                           bool init = false)
            : View(MemoryAllocator::template allocateMemory<typename View::value_type>(outputIndices.getNumElements(),
                                                                                       init),
                   outputIndices, sorted) {}

        /**
         * @param other A reference to an object of type `DenseScoreVectorAllocator` that should be copied
         */
        DenseScoreVectorAllocator(const DenseScoreVectorAllocator<View, MemoryAllocator>& other) : View(other) {
            throw std::runtime_error("Objects of type DenseScoreVectorAllocator cannot be copied");
        }

        /**
         * @param other A reference to an object of type `DenseScoreVectorAllocator` that should be moved
         */
        DenseScoreVectorAllocator(DenseScoreVectorAllocator<View, MemoryAllocator>&& other) : View(std::move(other)) {
            other.release();
        }

        virtual ~DenseScoreVectorAllocator() override {
            MemoryAllocator::freeMemory(View::array);
        }
};

/**
 * An one-dimensional vector that stores the scores that may be predicted by a rule, as well as an overall quality
 * score that assesses the overall quality of the rule, in a C-contiguous array.
 *
 * @tparam ScoreType   The type of the predicted scores
 * @tparam IndexVector The type of the vector that provides access to the indices of the outputs for which the rule may
 *                     predict
 */
template<typename ScoreType, typename IndexVector>
class DenseScoreVector final
    : public ViewDecorator<DenseScoreVectorAllocator<DenseScoreVectorView<ScoreType, IndexVector>>>,
      virtual public IScoreVector {
    public:

        /**
         * @param outputIndices A reference to an object of template type `IndexVector` that provides access to the
         *                      indices of the outputs for which the rule may predict
         * @param sorted        True, if the indices of the outputs for which the rule may predict are sorted in
         *                      increasing order, false otherwise
         */
        DenseScoreVector(const IndexVector& outputIndices, bool sorted);

        /**
         * An iterator that provides read-only access to the indices.
         */
        using index_const_iterator = IndexVector::const_iterator;

        /**
         * An iterator that provides access to the predicted scores and allows to modify them.
         */
        using value_iterator = View<ScoreType>::iterator;

        /**
         * An iterator that provides read-only access to the predicted scores.
         */
        using value_const_iterator = View<ScoreType>::const_iterator;

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
         * Returns a `value_iterator` to the beginning of the predicted scores.
         *
         * @return A `value_iterator` to the beginning
         */
        value_iterator values_begin();

        /**
         * Returns a `value_iterator` to the end of the predicted scores.
         *
         * @return A `value_iterator` to the end
         */
        value_iterator values_end();

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

        /**
         * Returns the number of outputs for which the rule may predict.
         *
         * @return The number of outputs
         */
        uint32 getNumElements() const;

        /**
         * Returns whether the rule may only predict for a subset of the available outputs, or not.
         *
         * @return True, if the rule may only predict for a subset of the available outputs, false otherwise
         */
        bool isPartial() const;

        /**
         * Sets the quality of the rule.
         *
         * @param quality The quality to be set
         */
        void setQuality(float64 quality);

        float64 getQuality() const override;

        void visit(BitVisitor<CompleteIndexVector> completeBitVisitor, BitVisitor<PartialIndexVector> partialBitVisitor,
                   DenseVisitor<float32, CompleteIndexVector> completeDense32BitVisitor,
                   DenseVisitor<float32, PartialIndexVector> partialDense32BitVisitor,
                   DenseVisitor<float64, CompleteIndexVector> completeDense64BitVisitor,
                   DenseVisitor<float64, PartialIndexVector> partialDense64BitVisitor,
                   DenseBinnedVisitor<float32, CompleteIndexVector> completeDenseBinned32BitVisitor,
                   DenseBinnedVisitor<float32, PartialIndexVector> partialDenseBinned32BitVisitor,
                   DenseBinnedVisitor<float64, CompleteIndexVector> completeDenseBinned64BitVisitor,
                   DenseBinnedVisitor<float64, PartialIndexVector> partialDenseBinned64BitVisitor) const override;
};
