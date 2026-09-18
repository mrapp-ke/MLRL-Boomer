/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/iterator/iterator_binned.hpp"
#include "mlrl/common/rule_evaluation/score_vector_decorator.hpp"
#include "mlrl/common/util/quality.hpp"

/**
 * An one dimensional view that provides access to scores that may be predicted by a rule and are stored in
 * pre-allocated arrays of a specific size, as well as an overall quality score that assesses the quality of the rule.
 * The predicted scores correspond to bins for which the same prediction is made.
 *
 * @tparam ScoreType   The type of the predicted scores
 * @tparam IndexVector The type of the vector that provides access to the indices of the outputs for which the rule may
 *                     predict
 */
template<typename ScoreType, typename IndexVector>
class MLRLCOMMON_API DenseBinnedScoreVectorView : public Quality {
    protected:

        /**
         * A view that provides access to the indices of the bins, predictions for individual outputs correspond to.
         */
        View<uint32> binIndices_;

        /**
         * A view that provides access to the predicted scores that correspond to individual bins.
         */
        Vector<ScoreType> binValues_;

    private:

        const IndexVector& outputIndices_;

        const bool sorted_;

    public:

        /**
         * @param binIndices    A pointer to an array of type `uint32` that stores the indices of the bins, the
         *                      predictions for individual outputs correspond to
         * @param binValues     A pointer to an array of template type `ScoreType` that stores the predicted scores that
         *                      correspond to individual bins
         * @param outputIndices A reference to an object of template type `IndexVector` that provides access to the
         *                      indices of the outputs for which the rule may predict
         * @param numBins       The number of bins
         * @param sorted        True, if the indices of the outputs for which the rule may predict are sorted in
         *                      increasing order, false otherwise
         */
        explicit DenseBinnedScoreVectorView(uint32* binIndices, ScoreType* binValues, const IndexVector& outputIndices,
                                            uint32 numBins, bool sorted)
            : binIndices_(binIndices, outputIndices.getNumElements()), binValues_(binValues, numBins),
              outputIndices_(outputIndices), sorted_(sorted) {}

        /**
         * @param other A const reference to an object of type `DenseBinnedScoreVectorView` that should be copied
         */
        DenseBinnedScoreVectorView(const DenseBinnedScoreVectorView<ScoreType, IndexVector>& other)
            : binIndices_(other.binIndices_), binValues_(other.binValues_), outputIndices_(other.outputIndices_),
              sorted_(other.sorted_) {}

        /**
         * @param other A reference to an object of type `DenseBinnedScoreVectorView` that should be moved
         */
        DenseBinnedScoreVectorView(DenseBinnedScoreVectorView<ScoreType, IndexVector>&& other)
            : binIndices_(std::move(other.binIndices_)), binValues_(std::move(other.binValues_)),
              outputIndices_(other.outputIndices_), sorted_(other.sorted_) {}

        virtual ~DenseBinnedScoreVectorView() override {}

        /**
         * The type of the vector that provides access to the indices of the outputs for which the rule may predict.
         */
        using index_vector_type = IndexVector;

        /**
         * The type of the predicted scores.
         */
        using score_type = ScoreType;

        /**
         * An iterator that provides read-only access to the indices of the output for which the rule predicts.
         */
        using index_const_iterator = IndexVector::const_iterator;

        /**
         * An iterator that provides read-only access to the predicted scores that correspond to individual outputs.
         */
        using value_const_iterator = BinnedIterator<const ScoreType>;

        /**
         * An iterator that provides access to the indices that correspond to individual bins and allows to modify them.
         */
        using bin_index_iterator = View<uint32>::iterator;

        /**
         * An iterator that provides read-only access to the indices that correspond to individual bins.
         */
        using bin_index_const_iterator = View<uint32>::const_iterator;

        /**
         * An iterator that provides access to the predicted scores that correspond to individual bins and allows to
         * modify them.
         */
        using bin_value_iterator = View<ScoreType>::iterator;

        /**
         * An iterator that provides read-only access to the predicted scores that correspond to individual bins.
         */
        using bin_value_const_iterator = View<ScoreType>::const_iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the indices that correspond to individual outputs.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const {
            return outputIndices_.cbegin();
        }

        /**
         * Returns an `index_const_iterator` to the end of the indices that correspond to individual outputs.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const {
            return outputIndices_.cend();
        }

        /**
         * Returns a `value_const_iterator` to the beginning of the predicted scores that correspond to individual
         * outputs.
         *
         * @return A `value_const_iterator` to the beginning
         */
        value_const_iterator cbegin() const {
            return value_const_iterator(View<const uint32>(this->bin_indices_cbegin()),
                                        View<const ScoreType>(this->bin_values_cbegin()), 0);
        }

        /**
         * Returns a `value_const_iterator` to the end of the predicted scores that correspond to individual outputs.
         *
         * @return A `value_const_iterator` to the end
         */
        value_const_iterator cend() const {
            return value_const_iterator(View<const uint32>(this->bin_indices_cbegin()),
                                        View<const ScoreType>(this->bin_values_cbegin()), this->getNumElements());
        }

        /**
         * Returns an `bin_index_iterator` to the beginning of the indices that correspond to individual bins.
         *
         * @return An `bin_index_iterator` to the beginning
         */
        bin_index_iterator bin_indices_begin() {
            return binIndices_.begin();
        }

        /**
         * Returns an `bin_index_iterator` to the end of the indices that correspond to individual bins.
         *
         * @return An `bin_index_iterator` to the end
         */
        bin_index_iterator bin_indices_end() {
            return &binIndices_.array[outputIndices_.getNumElements()];
        }

        /**
         * Returns an `bin_index_const_iterator` to the beginning of the indices that correspond to individual bins.
         *
         * @return An `bin_index_const_iterator` to the beginning
         */
        bin_index_const_iterator bin_indices_cbegin() const {
            return binIndices_.cbegin();
        }

        /**
         * Returns an `bin_index_const_iterator` to the end of the indices that correspond to individual bins.
         *
         * @return An `bin_index_const_iterator` to the end
         */
        bin_index_const_iterator bin_indices_cend() const {
            return &binIndices_.array[outputIndices_.getNumElements()];
        }

        /**
         * Returns a `bin_value_iterator` to the beginning of the predicted scores that correspond to individual bins.
         *
         * @return A `bin_value_iterator` to the beginning
         */
        bin_value_iterator bin_values_begin() {
            return binValues_.begin();
        }

        /**
         * Returns a `bin_value_iterator` to the end of the predicted scores that correspond to individual bins.
         *
         * @return A `bin_value_iterator` to the end
         */
        bin_value_iterator bin_values_end() {
            return binValues_.end();
        }

        /**
         * Returns a `bin_value_const_iterator` to the beginning of the predicted scores that correspond to individual
         * bins.
         *
         * @return A `bin_value_const_iterator` to the beginning
         */
        bin_value_const_iterator bin_values_cbegin() const {
            return binValues_.cbegin();
        }

        /**
         * Returns a `bin_value_const_iterator` to the end of the predicted scores that correspond to individual bins.
         *
         * @return A `bin_value_const_iterator` to the end
         */
        bin_value_const_iterator bin_values_cend() const {
            return binValues_.cend();
        }

        /**
         * Returns the number of elements in the view.
         *
         * @return The number of elements
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
 * Allocates the memory, a `DenseBinnedScoreVectorView` provides access to.
 *
 * @tparam View             The type of the view
 * @tparam MemoryAllocator  The type of the memory allocator to be used
 */
template<typename View, typename MemoryAllocator = DefaultMemoryAllocator>
class MLRLCOMMON_API DenseBinnedScoreVectorAllocator : public View {
    private:

        uint32 maxCapacity_;

    public:

        /**
         * @param outputIndices A reference to an object of template type `View::index_vector_type` that provides access
         *                      to the indices of the outputs for which the rule may predict
         * @param numBins       The number of bins
         * @param sorted        True, if the indices of the outputs for which the rule may predict are sorted in
         *                      increasing order, false otherwise
         * @param init          True, if all elements in the view should be value-initialized, false otherwise
         */
        explicit DenseBinnedScoreVectorAllocator(const typename View::index_vector_type& outputIndices, uint32 numBins,
                                                 bool sorted, bool init = false)
            : View(MemoryAllocator::template allocateMemory<uint32>(numBins, init),
                   MemoryAllocator::template allocateMemory<typename View::score_type>(outputIndices.getNumElements(),
                                                                                       init),
                   outputIndices, numBins, sorted) {}

        /**
         * @param other A reference to an object of type `DenseBinnedScoreVectorAllocator` that should be copied
         */
        DenseBinnedScoreVectorAllocator(const DenseBinnedScoreVectorAllocator<View, MemoryAllocator>& other) = delete;

        /**
         * @param other A reference to an object of type `DenseBinnedScoreVectorAllocator` that should be moved
         */
        DenseBinnedScoreVectorAllocator(DenseBinnedScoreVectorAllocator<View, MemoryAllocator>&& other)
            : View(std::move(other)) {
            other.binIndices_.release();
            other.binValues_.release();
        }

        virtual ~DenseBinnedScoreVectorAllocator() override {
            MemoryAllocator::freeMemory(View::binIndices_.array);
            MemoryAllocator::freeMemory(View::binValues_.array);
        }

        /**
         * Resizes the view by re-allocating the memory it provides access to.
         *
         * @param numBins       The number of bins to which the view should be resized
         * @param freeMemory    True, if unused memory should be freed, false otherwise
         */
        void resize(uint32 numBins, bool freeMemory) {
            if (numBins < maxCapacity_) {
                if (freeMemory) {
                    View::binValues_.array =
                      MemoryAllocator::reallocateMemory(View::binValues_.array, View::binValues_.numElements, numBins);
                    maxCapacity_ = numBins;
                }
            } else if (numBins > maxCapacity_) {
                View::binValues_.array =
                  MemoryAllocator::reallocateMemory(View::binValues_.array, View::binValues_.numElements, numBins);
                maxCapacity_ = numBins;
            }

            View::binValues_.numElements = numBins;
        }
};

/**
 * An one dimensional vector that stores the scores that may be predicted by a rule, as well as an overall quality score
 * that assesses the quality of the rule, in C-contiguous arrays. The predicted scores correspond to bins for which the
 * same prediction is made,
 *
 * @tparam ScoreType   The type of the predicted scores
 * @tparam IndexVector The type of the vector that provides access to the indices of the outputs for which the rule may
 *                     predict
 */
template<typename ScoreType, typename IndexVector>
class DenseBinnedScoreVector final
    : public AbstractScoreVectorViewDecorator<
        DenseBinnedScoreVectorAllocator<DenseBinnedScoreVectorView<ScoreType, IndexVector>>> {
    public:

        /**
         * @param outputIndices A reference to an object of template type `IndexVector` that provides access to the
         *                      indices of the outputs for which the rule may predict
         * @param numBins       The number of bins
         * @param sorted        True, if the indices of the outputs for which the rule may predict are sorted in
         *                      increasing order, false otherwise
         */
        DenseBinnedScoreVector(const IndexVector& outputIndices, uint32 numBins, bool sorted);

        /**
         * An iterator that provides read-only access to the indices of the output for which the rule predicts.
         */
        using index_const_iterator = IndexVector::const_iterator;

        /**
         * An iterator that provides read-only access to the predicted scores that correspond to individual outputs.
         */
        using value_const_iterator = BinnedIterator<const ScoreType>;

        /**
         * An iterator that provides access to the indices that correspond to individual bins and allows to modify them.
         */
        using bin_index_iterator = View<uint32>::iterator;

        /**
         * An iterator that provides read-only access to the indices that correspond to individual bins.
         */
        using bin_index_const_iterator = View<uint32>::const_iterator;

        /**
         * An iterator that provides access to the predicted scores that correspond to individual bins and allows to
         * modify them.
         */
        using bin_value_iterator = View<ScoreType>::iterator;

        /**
         * An iterator that provides read-only access to the predicted scores that correspond to individual bins.
         */
        using bin_value_const_iterator = View<ScoreType>::const_iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the indices that correspond to individual outputs.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the indices that correspond to individual outputs.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const;

        /**
         * Returns a `value_const_iterator` to the beginning of the predicted scores that correspond to individual
         * outputs.
         *
         * @return A `value_const_iterator` to the beginning
         */
        value_const_iterator cbegin() const;

        /**
         * Returns a `value_const_iterator` to the end of the predicted scores that correspond to individual outputs.
         *
         * @return A `value_const_iterator` to the end
         */
        value_const_iterator cend() const;

        /**
         * Returns an `bin_index_iterator` to the beginning of the indices that correspond to individual bins.
         *
         * @return An `bin_index_iterator` to the beginning
         */
        bin_index_iterator bin_indices_begin();

        /**
         * Returns an `bin_index_iterator` to the end of the indices that correspond to individual bins.
         *
         * @return An `bin_index_iterator` to the end
         */
        bin_index_iterator bin_indices_end();

        /**
         * Returns an `bin_index_const_iterator` to the beginning of the indices that correspond to individual bins.
         *
         * @return An `bin_index_const_iterator` to the beginning
         */
        bin_index_const_iterator bin_indices_cbegin() const;

        /**
         * Returns an `bin_index_const_iterator` to the end of the indices that correspond to individual bins.
         *
         * @return An `bin_index_const_iterator` to the end
         */
        bin_index_const_iterator bin_indices_cend() const;

        /**
         * Returns a `bin_value_iterator` to the beginning of the predicted scores that correspond to individual bins.
         *
         * @return A `bin_value_iterator` to the beginning
         */
        bin_value_iterator bin_values_begin();

        /**
         * Returns a `bin_value_iterator` to the end of the predicted scores that correspond to individual bins.
         *
         * @return A `bin_value_iterator` to the end
         */
        bin_value_iterator bin_values_end();

        /**
         * Returns a `bin_value_const_iterator` to the beginning of the predicted scores that correspond to individual
         * bins.
         *
         * @return A `bin_value_const_iterator` to the beginning
         */
        bin_value_const_iterator bin_values_cbegin() const;

        /**
         * Returns a `bin_value_const_iterator` to the end of the predicted scores that correspond to individual bins.
         *
         * @return A `bin_value_const_iterator` to the end
         */
        bin_value_const_iterator bin_values_cend() const;

        /**
         * Sets the number of bins in the vector.
         *
         * @param numBins       The number of bins to be set
         * @param freeMemory    True, if unused memory should be freed, if possible, false otherwise
         */
        void setNumBins(uint32 numBins, bool freeMemory);

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
