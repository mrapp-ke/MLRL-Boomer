#include "mlrl/boosting/data/statistic_vector_label_wise_sparse.hpp"

#include "mlrl/common/util/view_functions.hpp"
#include "statistic_vector_label_wise_sparse_common.hpp"

namespace boosting {

    SparseLabelWiseStatisticVector::ConstIterator::ConstIterator(Vector<Triple<float64>>::const_iterator iterator,
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
        : VectorDecorator<AllocatedVector<Triple<float64>>>(AllocatedVector<Triple<float64>>(numElements, init)),
          sumOfWeights_(0) {}

    SparseLabelWiseStatisticVector::SparseLabelWiseStatisticVector(const SparseLabelWiseStatisticVector& other)
        : SparseLabelWiseStatisticVector(other.view_.numElements) {
        copyView(other.view_.array, this->view_.array, this->view_.numElements);
        sumOfWeights_ = other.sumOfWeights_;
    }

    SparseLabelWiseStatisticVector::const_iterator SparseLabelWiseStatisticVector::cbegin() const {
        return ConstIterator(this->view_.array, sumOfWeights_);
    }

    SparseLabelWiseStatisticVector::const_iterator SparseLabelWiseStatisticVector::cend() const {
        return ConstIterator(&this->view_.array[this->view_.numElements], sumOfWeights_);
    }

    void SparseLabelWiseStatisticVector::clear() {
        sumOfWeights_ = 0;
        setViewToZeros(this->view_.array, this->view_.numElements);
    }

    void SparseLabelWiseStatisticVector::add(const SparseLabelWiseStatisticVector& vector) {
        sumOfWeights_ += vector.sumOfWeights_;
        addToView(this->view_.array, vector.view_.array, this->view_.numElements);
    }

    void SparseLabelWiseStatisticVector::add(const SparseLabelWiseStatisticConstView& view, uint32 row) {
        sumOfWeights_ += 1;
        addToSparseLabelWiseStatisticVector(this->view_.array, view.cbegin(row), view.cend(row));
    }

    void SparseLabelWiseStatisticVector::add(const SparseLabelWiseStatisticConstView& view, uint32 row,
                                             float64 weight) {
        if (weight != 0) {
            sumOfWeights_ += weight;
            addToSparseLabelWiseStatisticVector(this->view_.array, view.cbegin(row), view.cend(row), weight);
        }
    }

    void SparseLabelWiseStatisticVector::remove(const SparseLabelWiseStatisticConstView& view, uint32 row) {
        sumOfWeights_ -= 1;
        removeFromSparseLabelWiseStatisticVector(this->view_.array, view.cbegin(row), view.cend(row));
    }

    void SparseLabelWiseStatisticVector::remove(const SparseLabelWiseStatisticConstView& view, uint32 row,
                                                float64 weight) {
        if (weight != 0) {
            sumOfWeights_ -= weight;
            removeFromSparseLabelWiseStatisticVector(this->view_.array, view.cbegin(row), view.cend(row), weight);
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseStatisticConstView& view, uint32 row,
                                                     const CompleteIndexVector& indices) {
        sumOfWeights_ += 1;
        addToSparseLabelWiseStatisticVector(this->view_.array, view.cbegin(row), view.cend(row));
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseStatisticConstView& view, uint32 row,
                                                     const PartialIndexVector& indices) {
        sumOfWeights_ += 1;
        SparseLabelWiseStatisticConstView::const_row viewRow = view[row];
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        uint32 numElements = indices.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];
            const IndexedValue<Tuple<float64>>* entry = viewRow[index];

            if (entry) {
                const Tuple<float64>& tuple = entry->value;
                Triple<float64>& triple = this->view_.array[i];
                triple.first += (tuple.first);
                triple.second += (tuple.second);
                triple.third += 1;
            }
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseStatisticConstView& view, uint32 row,
                                                     const CompleteIndexVector& indices, float64 weight) {
        if (weight != 0) {
            sumOfWeights_ += weight;
            addToSparseLabelWiseStatisticVector(this->view_.array, view.cbegin(row), view.cend(row), weight);
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseStatisticConstView& view, uint32 row,
                                                     const PartialIndexVector& indices, float64 weight) {
        if (weight != 0) {
            sumOfWeights_ += weight;
            SparseLabelWiseStatisticConstView::const_row viewRow = view[row];
            PartialIndexVector::const_iterator indexIterator = indices.cbegin();
            uint32 numElements = indices.getNumElements();

            for (uint32 i = 0; i < numElements; i++) {
                uint32 index = indexIterator[i];
                const IndexedValue<Tuple<float64>>* entry = viewRow[index];

                if (entry) {
                    const Tuple<float64>& tuple = entry->value;
                    Triple<float64>& triple = this->view_.array[i];
                    triple.first += (tuple.first * weight);
                    triple.second += (tuple.second * weight);
                    triple.third += weight;
                }
            }
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseHistogramConstView& view, uint32 row,
                                                     const CompleteIndexVector& indices) {
        SparseLabelWiseHistogramConstView::weight_const_iterator weightIterator = view.weights_cbegin();
        float64 binWeight = weightIterator[row];

        if (binWeight != 0) {
            sumOfWeights_ += binWeight;
            addToView(this->view_.array, view.cbegin(row), this->view_.numElements);
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseHistogramConstView& view, uint32 row,
                                                     const PartialIndexVector& indices) {
        SparseLabelWiseHistogramConstView::weight_const_iterator weightIterator = view.weights_cbegin();
        float64 binWeight = weightIterator[row];

        if (binWeight != 0) {
            sumOfWeights_ += binWeight;
            addToView(this->view_.array, view.cbegin(row), indices.cbegin(), indices.getNumElements());
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseHistogramConstView& view, uint32 row,
                                                     const CompleteIndexVector& indices, float64 weight) {
        SparseLabelWiseHistogramConstView::weight_const_iterator weightIterator = view.weights_cbegin();
        float64 binWeight = weightIterator[row] * weight;

        if (binWeight != 0) {
            sumOfWeights_ += binWeight;
            addToView(this->view_.array, view.cbegin(row), this->view_.numElements, weight);
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseHistogramConstView& view, uint32 row,
                                                     const PartialIndexVector& indices, float64 weight) {
        SparseLabelWiseHistogramConstView::weight_const_iterator weightIterator = view.weights_cbegin();
        float64 binWeight = weightIterator[row] * weight;

        if (binWeight != 0) {
            sumOfWeights_ += binWeight;
            addToView(this->view_.array, view.cbegin(row), indices.cbegin(), indices.getNumElements(), weight);
        }
    }

    void SparseLabelWiseStatisticVector::difference(const SparseLabelWiseStatisticVector& first,
                                                    const CompleteIndexVector& firstIndices,
                                                    const SparseLabelWiseStatisticVector& second) {
        sumOfWeights_ = first.sumOfWeights_ - second.sumOfWeights_;
        setViewToDifference(this->view_.array, first.view_.array, second.view_.array, this->view_.numElements);
    }

    void SparseLabelWiseStatisticVector::difference(const SparseLabelWiseStatisticVector& first,
                                                    const PartialIndexVector& firstIndices,
                                                    const SparseLabelWiseStatisticVector& second) {
        sumOfWeights_ = first.sumOfWeights_ - second.sumOfWeights_;
        setViewToDifference(this->view_.array, first.view_.array, second.view_.array, firstIndices.cbegin(),
                            this->view_.numElements);
    }

}
