#include "mlrl/boosting/data/statistic_vector_label_wise_sparse.hpp"

#include "statistic_vector_label_wise_sparse_common.hpp"

namespace boosting {

    SparseLabelWiseStatisticVector::ConstIterator::ConstIterator(View<Triple<float64>>::const_iterator iterator,
                                                                 float64 sumOfWeights)
        : iterator_(iterator), sumOfWeights_(sumOfWeights) {}

    SparseLabelWiseStatisticVector::ConstIterator::value_type SparseLabelWiseStatisticVector::ConstIterator::operator[](
      uint32 index) const {
        const Triple<float64>& triple = iterator_[index];
        float64 gradient = triple.first;
        float64 hessian = triple.second + (sumOfWeights_ - triple.third);
        return Tuple<float64>(gradient, hessian);
    }

    SparseLabelWiseStatisticVector::ConstIterator::value_type SparseLabelWiseStatisticVector::ConstIterator::operator*()
      const {
        const Triple<float64>& triple = *iterator_;
        float64 gradient = triple.first;
        float64 hessian = triple.second + (sumOfWeights_ - triple.third);
        return Tuple<float64>(gradient, hessian);
    }

    SparseLabelWiseStatisticVector::ConstIterator& SparseLabelWiseStatisticVector::ConstIterator::operator++() {
        ++iterator_;
        return *this;
    }

    SparseLabelWiseStatisticVector::ConstIterator& SparseLabelWiseStatisticVector::ConstIterator::operator++(int n) {
        iterator_++;
        return *this;
    }

    SparseLabelWiseStatisticVector::ConstIterator& SparseLabelWiseStatisticVector::ConstIterator::operator--() {
        --iterator_;
        return *this;
    }

    SparseLabelWiseStatisticVector::ConstIterator& SparseLabelWiseStatisticVector::ConstIterator::operator--(int n) {
        iterator_--;
        return *this;
    }

    bool SparseLabelWiseStatisticVector::ConstIterator::operator!=(const ConstIterator& rhs) const {
        return iterator_ != rhs.iterator_;
    }

    bool SparseLabelWiseStatisticVector::ConstIterator::operator==(const ConstIterator& rhs) const {
        return iterator_ == rhs.iterator_;
    }

    SparseLabelWiseStatisticVector::ConstIterator::difference_type
      SparseLabelWiseStatisticVector::ConstIterator::operator-(const ConstIterator& rhs) const {
        return iterator_ - rhs.iterator_;
    }

    SparseLabelWiseStatisticVector::SparseLabelWiseStatisticVector(uint32 numElements, bool init)
        : ClearableViewDecorator<VectorDecorator<AllocatedVector<Triple<float64>>>>(
          AllocatedVector<Triple<float64>>(numElements, init)),
          sumOfWeights_(0) {}

    SparseLabelWiseStatisticVector::SparseLabelWiseStatisticVector(const SparseLabelWiseStatisticVector& other)
        : SparseLabelWiseStatisticVector(other.getNumElements()) {
        copyView(other.view.cbegin(), this->view.begin(), this->getNumElements());
        sumOfWeights_ = other.sumOfWeights_;
    }

    SparseLabelWiseStatisticVector::const_iterator SparseLabelWiseStatisticVector::cbegin() const {
        return ConstIterator(this->view.cbegin(), sumOfWeights_);
    }

    SparseLabelWiseStatisticVector::const_iterator SparseLabelWiseStatisticVector::cend() const {
        return ConstIterator(this->view.cend(), sumOfWeights_);
    }

    void SparseLabelWiseStatisticVector::add(const SparseLabelWiseStatisticVector& vector) {
        sumOfWeights_ += vector.sumOfWeights_;
        addToView(this->view.begin(), vector.view.cbegin(), this->getNumElements());
    }

    void SparseLabelWiseStatisticVector::add(const SparseSetView<Tuple<float64>>& view, uint32 row) {
        sumOfWeights_ += 1;
        addToSparseLabelWiseStatisticVector(this->view.begin(), view.cbegin(row), view.cend(row));
    }

    void SparseLabelWiseStatisticVector::add(const SparseSetView<Tuple<float64>>& view, uint32 row, float64 weight) {
        if (weight != 0) {
            sumOfWeights_ += weight;
            addToSparseLabelWiseStatisticVector(this->view.begin(), view.cbegin(row), view.cend(row), weight);
        }
    }

    void SparseLabelWiseStatisticVector::remove(const SparseSetView<Tuple<float64>>& view, uint32 row) {
        sumOfWeights_ -= 1;
        removeFromSparseLabelWiseStatisticVector(this->view.begin(), view.cbegin(row), view.cend(row));
    }

    void SparseLabelWiseStatisticVector::remove(const SparseSetView<Tuple<float64>>& view, uint32 row, float64 weight) {
        if (weight != 0) {
            sumOfWeights_ -= weight;
            removeFromSparseLabelWiseStatisticVector(this->view.begin(), view.cbegin(row), view.cend(row), weight);
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseSetView<Tuple<float64>>& view, uint32 row,
                                                     const CompleteIndexVector& indices) {
        sumOfWeights_ += 1;
        addToSparseLabelWiseStatisticVector(this->view.begin(), view.cbegin(row), view.cend(row));
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseSetView<Tuple<float64>>& view, uint32 row,
                                                     const PartialIndexVector& indices) {
        sumOfWeights_ += 1;
        SparseSetView<Tuple<float64>>::const_row viewRow = view[row];
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        uint32 numElements = indices.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];
            const IndexedValue<Tuple<float64>>* entry = viewRow[index];

            if (entry) {
                const Tuple<float64>& tuple = entry->value;
                Triple<float64>& triple = this->view.begin()[i];
                triple.first += (tuple.first);
                triple.second += (tuple.second);
                triple.third += 1;
            }
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseSetView<Tuple<float64>>& view, uint32 row,
                                                     const CompleteIndexVector& indices, float64 weight) {
        if (weight != 0) {
            sumOfWeights_ += weight;
            addToSparseLabelWiseStatisticVector(this->view.begin(), view.cbegin(row), view.cend(row), weight);
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseSetView<Tuple<float64>>& view, uint32 row,
                                                     const PartialIndexVector& indices, float64 weight) {
        if (weight != 0) {
            sumOfWeights_ += weight;
            SparseSetView<Tuple<float64>>::const_row viewRow = view[row];
            PartialIndexVector::const_iterator indexIterator = indices.cbegin();
            uint32 numElements = indices.getNumElements();

            for (uint32 i = 0; i < numElements; i++) {
                uint32 index = indexIterator[i];
                const IndexedValue<Tuple<float64>>* entry = viewRow[index];

                if (entry) {
                    const Tuple<float64>& tuple = entry->value;
                    Triple<float64>& triple = this->view.begin()[i];
                    triple.first += (tuple.first * weight);
                    triple.second += (tuple.second * weight);
                    triple.third += weight;
                }
            }
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseHistogramView& view, uint32 row,
                                                     const CompleteIndexVector& indices) {
        SparseLabelWiseHistogramView::weight_const_iterator weightIterator = view.weights_cbegin();
        float64 binWeight = weightIterator[row];

        if (binWeight != 0) {
            sumOfWeights_ += binWeight;
            addToView(this->view.begin(), view.cbegin(row), this->getNumElements());
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseHistogramView& view, uint32 row,
                                                     const PartialIndexVector& indices) {
        SparseLabelWiseHistogramView::weight_const_iterator weightIterator = view.weights_cbegin();
        float64 binWeight = weightIterator[row];

        if (binWeight != 0) {
            sumOfWeights_ += binWeight;
            addToView(this->view.begin(), view.cbegin(row), indices.cbegin(), indices.getNumElements());
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseHistogramView& view, uint32 row,
                                                     const CompleteIndexVector& indices, float64 weight) {
        SparseLabelWiseHistogramView::weight_const_iterator weightIterator = view.weights_cbegin();
        float64 binWeight = weightIterator[row] * weight;

        if (binWeight != 0) {
            sumOfWeights_ += binWeight;
            addToView(this->view.begin(), view.cbegin(row), this->getNumElements(), weight);
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseHistogramView& view, uint32 row,
                                                     const PartialIndexVector& indices, float64 weight) {
        SparseLabelWiseHistogramView::weight_const_iterator weightIterator = view.weights_cbegin();
        float64 binWeight = weightIterator[row] * weight;

        if (binWeight != 0) {
            sumOfWeights_ += binWeight;
            addToView(this->view.begin(), view.cbegin(row), indices.cbegin(), indices.getNumElements(), weight);
        }
    }

    void SparseLabelWiseStatisticVector::difference(const SparseLabelWiseStatisticVector& first,
                                                    const CompleteIndexVector& firstIndices,
                                                    const SparseLabelWiseStatisticVector& second) {
        sumOfWeights_ = first.sumOfWeights_ - second.sumOfWeights_;
        setViewToDifference(this->view.begin(), first.view.cbegin(), second.view.cbegin(), this->getNumElements());
    }

    void SparseLabelWiseStatisticVector::difference(const SparseLabelWiseStatisticVector& first,
                                                    const PartialIndexVector& firstIndices,
                                                    const SparseLabelWiseStatisticVector& second) {
        sumOfWeights_ = first.sumOfWeights_ - second.sumOfWeights_;
        setViewToDifference(this->view.begin(), first.view.cbegin(), second.view.cbegin(), firstIndices.cbegin(),
                            this->getNumElements());
    }

    void SparseLabelWiseStatisticVector::clear() {
        ClearableViewDecorator<VectorDecorator<AllocatedVector<Triple<float64>>>>::clear();
        sumOfWeights_ = 0;
    }

}
