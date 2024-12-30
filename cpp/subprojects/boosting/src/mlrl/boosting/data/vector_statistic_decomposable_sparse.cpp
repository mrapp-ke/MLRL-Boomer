#include "mlrl/boosting/data/vector_statistic_decomposable_sparse.hpp"

namespace boosting {

    static inline void addToSparseDecomposableStatisticVector(View<Triple<float64>>::iterator statistics,
                                                              SparseSetView<Tuple<float64>>::value_const_iterator begin,
                                                              SparseSetView<Tuple<float64>>::value_const_iterator end) {
        uint32 numElements = end - begin;

        for (uint32 i = 0; i < numElements; i++) {
            const IndexedValue<Tuple<float64>>& entry = begin[i];
            const Tuple<float64>& tuple = entry.value;
            Triple<float64>& triple = statistics[entry.index];
            triple.first += tuple.first;
            triple.second += tuple.second;
            triple.third += 1;
        }
    }

    static inline void addToSparseDecomposableStatisticVectorWeighted(
      View<Triple<float64>>::iterator statistics, SparseSetView<Tuple<float64>>::value_const_iterator begin,
      SparseSetView<Tuple<float64>>::value_const_iterator end, uint32 weight) {
        uint32 numElements = end - begin;

        for (uint32 i = 0; i < numElements; i++) {
            const IndexedValue<Tuple<float64>>& entry = begin[i];
            const Tuple<float64>& tuple = entry.value;
            Triple<float64>& triple = statistics[entry.index];
            triple.first += (tuple.first * weight);
            triple.second += (tuple.second * weight);
            triple.third += weight;
        }
    }

    static inline void removeFromSparseDecomposableStatisticVector(
      View<Triple<float64>>::iterator statistics, SparseSetView<Tuple<float64>>::value_const_iterator begin,
      SparseSetView<Tuple<float64>>::value_const_iterator end) {
        uint32 numElements = end - begin;

        for (uint32 i = 0; i < numElements; i++) {
            const IndexedValue<Tuple<float64>>& entry = begin[i];
            const Tuple<float64>& tuple = entry.value;
            Triple<float64>& triple = statistics[entry.index];
            triple.first -= tuple.first;
            triple.second -= tuple.second;
            triple.third -= 1;
        }
    }

    static inline void removeFromSparseDecomposableStatisticVectorWeighted(
      View<Triple<float64>>::iterator statistics, SparseSetView<Tuple<float64>>::value_const_iterator begin,
      SparseSetView<Tuple<float64>>::value_const_iterator end, uint32 weight) {
        uint32 numElements = end - begin;

        for (uint32 i = 0; i < numElements; i++) {
            const IndexedValue<Tuple<float64>>& entry = begin[i];
            const Tuple<float64>& tuple = entry.value;
            Triple<float64>& triple = statistics[entry.index];
            triple.first -= (tuple.first * weight);
            triple.second -= (tuple.second * weight);
            triple.third -= weight;
        }
    }

    SparseDecomposableStatisticVector::ConstIterator::ConstIterator(View<Triple<float64>>::const_iterator iterator,
                                                                    float64 sumOfWeights)
        : iterator_(iterator), sumOfWeights_(sumOfWeights) {}

    SparseDecomposableStatisticVector::ConstIterator::value_type
      SparseDecomposableStatisticVector::ConstIterator::operator[](uint32 index) const {
        const Triple<float64>& triple = iterator_[index];
        float64 gradient = triple.first;
        float64 hessian = triple.second + (sumOfWeights_ - triple.third);
        return Tuple<float64>(gradient, hessian);
    }

    SparseDecomposableStatisticVector::ConstIterator::value_type
      SparseDecomposableStatisticVector::ConstIterator::operator*() const {
        const Triple<float64>& triple = *iterator_;
        float64 gradient = triple.first;
        float64 hessian = triple.second + (sumOfWeights_ - triple.third);
        return Tuple<float64>(gradient, hessian);
    }

    SparseDecomposableStatisticVector::ConstIterator& SparseDecomposableStatisticVector::ConstIterator::operator++() {
        ++iterator_;
        return *this;
    }

    SparseDecomposableStatisticVector::ConstIterator& SparseDecomposableStatisticVector::ConstIterator::operator++(
      int n) {
        iterator_++;
        return *this;
    }

    SparseDecomposableStatisticVector::ConstIterator& SparseDecomposableStatisticVector::ConstIterator::operator--() {
        --iterator_;
        return *this;
    }

    SparseDecomposableStatisticVector::ConstIterator& SparseDecomposableStatisticVector::ConstIterator::operator--(
      int n) {
        iterator_--;
        return *this;
    }

    bool SparseDecomposableStatisticVector::ConstIterator::operator!=(const ConstIterator& rhs) const {
        return iterator_ != rhs.iterator_;
    }

    bool SparseDecomposableStatisticVector::ConstIterator::operator==(const ConstIterator& rhs) const {
        return iterator_ == rhs.iterator_;
    }

    SparseDecomposableStatisticVector::ConstIterator::difference_type
      SparseDecomposableStatisticVector::ConstIterator::operator-(const ConstIterator& rhs) const {
        return iterator_ - rhs.iterator_;
    }

    SparseDecomposableStatisticVector::SparseDecomposableStatisticVector(uint32 numElements, bool init)
        : ClearableViewDecorator<VectorDecorator<AllocatedVector<Triple<float64>>>>(
            AllocatedVector<Triple<float64>>(numElements, init)),
          sumOfWeights_(0) {}

    SparseDecomposableStatisticVector::SparseDecomposableStatisticVector(const SparseDecomposableStatisticVector& other)
        : SparseDecomposableStatisticVector(other.getNumElements()) {
        util::copyView(other.view.cbegin(), this->view.begin(), this->getNumElements());
        sumOfWeights_ = other.sumOfWeights_;
    }

    SparseDecomposableStatisticVector::const_iterator SparseDecomposableStatisticVector::cbegin() const {
        return ConstIterator(this->view.cbegin(), sumOfWeights_);
    }

    SparseDecomposableStatisticVector::const_iterator SparseDecomposableStatisticVector::cend() const {
        return ConstIterator(this->view.cend(), sumOfWeights_);
    }

    void SparseDecomposableStatisticVector::add(const SparseDecomposableStatisticVector& vector) {
        sumOfWeights_ += vector.sumOfWeights_;
        util::addToView(this->view.begin(), vector.view.cbegin(), this->getNumElements());
    }

    void SparseDecomposableStatisticVector::add(const SparseSetView<Tuple<float64>>& view, uint32 row) {
        sumOfWeights_ += 1;
        addToSparseDecomposableStatisticVector(this->view.begin(), view.values_cbegin(row), view.values_cend(row));
    }

    void SparseDecomposableStatisticVector::add(const SparseSetView<Tuple<float64>>& view, uint32 row, uint32 weight) {
        if (!isEqualToZero(weight)) {
            sumOfWeights_ += weight;
            addToSparseDecomposableStatisticVectorWeighted(this->view.begin(), view.values_cbegin(row),
                                                           view.values_cend(row), weight);
        }
    }

    void SparseDecomposableStatisticVector::remove(const SparseSetView<Tuple<float64>>& view, uint32 row) {
        sumOfWeights_ -= 1;
        removeFromSparseDecomposableStatisticVector(this->view.begin(), view.values_cbegin(row), view.values_cend(row));
    }

    void SparseDecomposableStatisticVector::remove(const SparseSetView<Tuple<float64>>& view, uint32 row,
                                                   uint32 weight) {
        if (!isEqualToZero(weight)) {
            sumOfWeights_ -= weight;
            removeFromSparseDecomposableStatisticVectorWeighted(this->view.begin(), view.values_cbegin(row),
                                                                view.values_cend(row), weight);
        }
    }

    void SparseDecomposableStatisticVector::addToSubset(const SparseSetView<Tuple<float64>>& view, uint32 row,
                                                        const CompleteIndexVector& indices) {
        sumOfWeights_ += 1;
        addToSparseDecomposableStatisticVector(this->view.begin(), view.values_cbegin(row), view.values_cend(row));
    }

    void SparseDecomposableStatisticVector::addToSubset(const SparseSetView<Tuple<float64>>& view, uint32 row,
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

    void SparseDecomposableStatisticVector::addToSubset(const SparseSetView<Tuple<float64>>& view, uint32 row,
                                                        const CompleteIndexVector& indices, uint32 weight) {
        if (!isEqualToZero(weight)) {
            sumOfWeights_ += weight;
            addToSparseDecomposableStatisticVectorWeighted(this->view.begin(), view.values_cbegin(row),
                                                           view.values_cend(row), weight);
        }
    }

    void SparseDecomposableStatisticVector::addToSubset(const SparseSetView<Tuple<float64>>& view, uint32 row,
                                                        const PartialIndexVector& indices, uint32 weight) {
        if (!isEqualToZero(weight)) {
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

    void SparseDecomposableStatisticVector::difference(const SparseDecomposableStatisticVector& first,
                                                       const CompleteIndexVector& firstIndices,
                                                       const SparseDecomposableStatisticVector& second) {
        sumOfWeights_ = first.sumOfWeights_ - second.sumOfWeights_;
        util::setViewToDifference(this->view.begin(), first.view.cbegin(), second.view.cbegin(),
                                  this->getNumElements());
    }

    void SparseDecomposableStatisticVector::difference(const SparseDecomposableStatisticVector& first,
                                                       const PartialIndexVector& firstIndices,
                                                       const SparseDecomposableStatisticVector& second) {
        sumOfWeights_ = first.sumOfWeights_ - second.sumOfWeights_;
        util::setViewToDifference(this->view.begin(), first.view.cbegin(), second.view.cbegin(), firstIndices.cbegin(),
                                  this->getNumElements());
    }

    void SparseDecomposableStatisticVector::clear() {
        ClearableViewDecorator<VectorDecorator<AllocatedVector<Triple<float64>>>>::clear();
        sumOfWeights_ = 0;
    }

}
