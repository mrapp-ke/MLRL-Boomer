#include "mlrl/boosting/data/vector_statistic_decomposable_sparse.hpp"

namespace boosting {

    template<typename WeightType>
    SparseDecomposableStatisticVector<WeightType>::ConstIterator::ConstIterator(
      typename View<SparseStatistic<float64, WeightType>>::const_iterator iterator, WeightType sumOfWeights)
        : iterator_(iterator), sumOfWeights_(sumOfWeights) {}

    template<typename WeightType>
    typename SparseDecomposableStatisticVector<WeightType>::ConstIterator::value_type
      SparseDecomposableStatisticVector<WeightType>::ConstIterator::operator[](uint32 index) const {
        const SparseStatistic<float64, WeightType>& statistic = iterator_[index];
        float64 gradient = statistic.gradient;
        float64 hessian = statistic.hessian + (sumOfWeights_ - statistic.weight);
        return Statistic<float64>(gradient, hessian);
    }

    template<typename WeightType>
    typename SparseDecomposableStatisticVector<WeightType>::ConstIterator::value_type
      SparseDecomposableStatisticVector<WeightType>::ConstIterator::operator*() const {
        const SparseStatistic<float64, WeightType>& statistic = *iterator_;
        float64 gradient = statistic.gradient;
        float64 hessian = statistic.hessian + (sumOfWeights_ - statistic.weight);
        return Statistic<float64>(gradient, hessian);
    }

    template<typename WeightType>
    typename SparseDecomposableStatisticVector<WeightType>::ConstIterator&
      SparseDecomposableStatisticVector<WeightType>::ConstIterator::operator++() {
        ++iterator_;
        return *this;
    }

    template<typename WeightType>
    typename SparseDecomposableStatisticVector<WeightType>::ConstIterator&
      SparseDecomposableStatisticVector<WeightType>::ConstIterator::operator++(int n) {
        iterator_++;
        return *this;
    }

    template<typename WeightType>
    typename SparseDecomposableStatisticVector<WeightType>::ConstIterator&
      SparseDecomposableStatisticVector<WeightType>::ConstIterator::operator--() {
        --iterator_;
        return *this;
    }

    template<typename WeightType>
    typename SparseDecomposableStatisticVector<WeightType>::ConstIterator&
      SparseDecomposableStatisticVector<WeightType>::ConstIterator::operator--(int n) {
        iterator_--;
        return *this;
    }

    template<typename WeightType>
    bool SparseDecomposableStatisticVector<WeightType>::ConstIterator::operator!=(const ConstIterator& rhs) const {
        return iterator_ != rhs.iterator_;
    }

    template<typename WeightType>
    bool SparseDecomposableStatisticVector<WeightType>::ConstIterator::operator==(const ConstIterator& rhs) const {
        return iterator_ == rhs.iterator_;
    }

    template<typename WeightType>
    typename SparseDecomposableStatisticVector<WeightType>::ConstIterator::difference_type
      SparseDecomposableStatisticVector<WeightType>::ConstIterator::operator-(const ConstIterator& rhs) const {
        return iterator_ - rhs.iterator_;
    }

    template<typename WeightType>
    static inline void addToSparseDecomposableStatisticVector(
      typename View<SparseStatistic<float64, WeightType>>::iterator statistics,
      SparseSetView<Statistic<float64>>::value_const_iterator begin,
      SparseSetView<Statistic<float64>>::value_const_iterator end) {
        uint32 numElements = end - begin;

        for (uint32 i = 0; i < numElements; i++) {
            const IndexedValue<Statistic<float64>>& entry = begin[i];
            const Statistic<float64>& statistic = entry.value;
            SparseStatistic<float64, WeightType>& sparseStatistic = statistics[entry.index];
            sparseStatistic.gradient += statistic.gradient;
            sparseStatistic.hessian += statistic.hessian;
            sparseStatistic.weight += 1;
        }
    }

    template<typename WeightType>
    static inline void addToSparseDecomposableStatisticVectorWeighted(
      typename View<SparseStatistic<float64, WeightType>>::iterator statistics,
      SparseSetView<Statistic<float64>>::value_const_iterator begin,
      SparseSetView<Statistic<float64>>::value_const_iterator end, WeightType weight) {
        uint32 numElements = end - begin;

        for (uint32 i = 0; i < numElements; i++) {
            const IndexedValue<Statistic<float64>>& entry = begin[i];
            const Statistic<float64>& statistic = entry.value;
            SparseStatistic<float64, WeightType>& sparseStatistic = statistics[entry.index];
            sparseStatistic.gradient += (statistic.gradient * weight);
            sparseStatistic.hessian += (statistic.hessian * weight);
            sparseStatistic.weight += weight;
        }
    }

    template<typename WeightType>
    static inline void removeFromSparseDecomposableStatisticVector(
      typename View<SparseStatistic<float64, WeightType>>::iterator statistics,
      SparseSetView<Statistic<float64>>::value_const_iterator begin,
      SparseSetView<Statistic<float64>>::value_const_iterator end) {
        uint32 numElements = end - begin;

        for (uint32 i = 0; i < numElements; i++) {
            const IndexedValue<Statistic<float64>>& entry = begin[i];
            const Statistic<float64>& statistic = entry.value;
            SparseStatistic<float64, WeightType>& sparseStatistic = statistics[entry.index];
            sparseStatistic.gradient -= statistic.gradient;
            sparseStatistic.hessian -= statistic.hessian;
            sparseStatistic.weight -= 1;
        }
    }

    template<typename WeightType>
    static inline void removeFromSparseDecomposableStatisticVectorWeighted(
      typename View<SparseStatistic<float64, WeightType>>::iterator statistics,
      SparseSetView<Statistic<float64>>::value_const_iterator begin,
      SparseSetView<Statistic<float64>>::value_const_iterator end, WeightType weight) {
        uint32 numElements = end - begin;

        for (uint32 i = 0; i < numElements; i++) {
            const IndexedValue<Statistic<float64>>& entry = begin[i];
            const Statistic<float64>& statistic = entry.value;
            SparseStatistic<float64, WeightType>& sparseStatistic = statistics[entry.index];
            sparseStatistic.gradient -= (statistic.gradient * weight);
            sparseStatistic.hessian -= (statistic.hessian * weight);
            sparseStatistic.weight -= weight;
        }
    }

    template<typename WeightType>
    SparseDecomposableStatisticVector<WeightType>::SparseDecomposableStatisticVector(uint32 numElements, bool init)
        : VectorDecorator<AllocatedVector<SparseStatistic<float64, WeightType>>>(
            AllocatedVector<SparseStatistic<float64, WeightType>>(numElements, init)),
          sumOfWeights_(0) {}

    template<typename WeightType>
    SparseDecomposableStatisticVector<WeightType>::SparseDecomposableStatisticVector(
      const SparseDecomposableStatisticVector<WeightType>& other)
        : SparseDecomposableStatisticVector(other.getNumElements()) {
        util::copyView(other.view.cbegin(), this->view.begin(), this->getNumElements());
        sumOfWeights_ = other.sumOfWeights_;
    }

    template<typename WeightType>
    typename SparseDecomposableStatisticVector<WeightType>::const_iterator
      SparseDecomposableStatisticVector<WeightType>::cbegin() const {
        return ConstIterator(this->view.cbegin(), sumOfWeights_);
    }

    template<typename WeightType>
    typename SparseDecomposableStatisticVector<WeightType>::const_iterator
      SparseDecomposableStatisticVector<WeightType>::cend() const {
        return ConstIterator(this->view.cend(), sumOfWeights_);
    }

    template<typename WeightType>
    void SparseDecomposableStatisticVector<WeightType>::add(const SparseDecomposableStatisticVector& vector) {
        sumOfWeights_ += vector.sumOfWeights_;
        util::addToView(this->view.begin(), vector.view.cbegin(), this->getNumElements());
    }

    template<typename WeightType>
    void SparseDecomposableStatisticVector<WeightType>::add(const SparseSetView<Statistic<float64>>& view, uint32 row) {
        sumOfWeights_ += 1;
        addToSparseDecomposableStatisticVector<WeightType>(this->view.begin(), view.values_cbegin(row),
                                                           view.values_cend(row));
    }

    template<typename WeightType>
    void SparseDecomposableStatisticVector<WeightType>::add(const SparseSetView<Statistic<float64>>& view, uint32 row,
                                                            WeightType weight) {
        if (!isEqualToZero(weight)) {
            sumOfWeights_ += weight;
            addToSparseDecomposableStatisticVectorWeighted<WeightType>(this->view.begin(), view.values_cbegin(row),
                                                                       view.values_cend(row), weight);
        }
    }

    template<typename WeightType>
    void SparseDecomposableStatisticVector<WeightType>::remove(const SparseSetView<Statistic<float64>>& view,
                                                               uint32 row) {
        sumOfWeights_ -= 1;
        removeFromSparseDecomposableStatisticVector<WeightType>(this->view.begin(), view.values_cbegin(row),
                                                                view.values_cend(row));
    }

    template<typename WeightType>
    void SparseDecomposableStatisticVector<WeightType>::remove(const SparseSetView<Statistic<float64>>& view,
                                                               uint32 row, WeightType weight) {
        if (!isEqualToZero(weight)) {
            sumOfWeights_ -= weight;
            removeFromSparseDecomposableStatisticVectorWeighted<WeightType>(this->view.begin(), view.values_cbegin(row),
                                                                            view.values_cend(row), weight);
        }
    }

    template<typename WeightType>
    void SparseDecomposableStatisticVector<WeightType>::addToSubset(const SparseSetView<Statistic<float64>>& view,
                                                                    uint32 row, const CompleteIndexVector& indices) {
        sumOfWeights_ += 1;
        addToSparseDecomposableStatisticVector<WeightType>(this->view.begin(), view.values_cbegin(row),
                                                           view.values_cend(row));
    }

    template<typename WeightType>
    void SparseDecomposableStatisticVector<WeightType>::addToSubset(const SparseSetView<Statistic<float64>>& view,
                                                                    uint32 row, const PartialIndexVector& indices) {
        sumOfWeights_ += 1;
        SparseSetView<Statistic<float64>>::const_row viewRow = view[row];
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        uint32 numElements = indices.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];
            const IndexedValue<Statistic<float64>>* entry = viewRow[index];

            if (entry) {
                const Statistic<float64>& statistic = entry->value;
                SparseStatistic<float64, WeightType>& sparseStatistic = this->view.begin()[i];
                sparseStatistic.gradient += (statistic.gradient);
                sparseStatistic.hessian += (statistic.hessian);
                sparseStatistic.weight += 1;
            }
        }
    }

    template<typename WeightType>
    void SparseDecomposableStatisticVector<WeightType>::addToSubset(const SparseSetView<Statistic<float64>>& view,
                                                                    uint32 row, const CompleteIndexVector& indices,
                                                                    WeightType weight) {
        if (!isEqualToZero(weight)) {
            sumOfWeights_ += weight;
            addToSparseDecomposableStatisticVectorWeighted<WeightType>(this->view.begin(), view.values_cbegin(row),
                                                                       view.values_cend(row), weight);
        }
    }

    template<typename WeightType>
    void SparseDecomposableStatisticVector<WeightType>::addToSubset(const SparseSetView<Statistic<float64>>& view,
                                                                    uint32 row, const PartialIndexVector& indices,
                                                                    WeightType weight) {
        if (!isEqualToZero(weight)) {
            sumOfWeights_ += weight;
            SparseSetView<Statistic<float64>>::const_row viewRow = view[row];
            PartialIndexVector::const_iterator indexIterator = indices.cbegin();
            uint32 numElements = indices.getNumElements();

            for (uint32 i = 0; i < numElements; i++) {
                uint32 index = indexIterator[i];
                const IndexedValue<Statistic<float64>>* entry = viewRow[index];

                if (entry) {
                    const Statistic<float64>& statistic = entry->value;
                    SparseStatistic<float64, WeightType>& sparseStatistic = this->view.begin()[i];
                    sparseStatistic.gradient += (statistic.gradient * weight);
                    sparseStatistic.hessian += (statistic.hessian * weight);
                    sparseStatistic.weight += weight;
                }
            }
        }
    }

    template<typename WeightType>
    void SparseDecomposableStatisticVector<WeightType>::difference(
      const SparseDecomposableStatisticVector<WeightType>& first, const CompleteIndexVector& firstIndices,
      const SparseDecomposableStatisticVector<WeightType>& second) {
        sumOfWeights_ = first.sumOfWeights_ - second.sumOfWeights_;
        util::setViewToDifference(this->view.begin(), first.view.cbegin(), second.view.cbegin(),
                                  this->getNumElements());
    }

    template<typename WeightType>
    void SparseDecomposableStatisticVector<WeightType>::difference(
      const SparseDecomposableStatisticVector<WeightType>& first, const PartialIndexVector& firstIndices,
      const SparseDecomposableStatisticVector<WeightType>& second) {
        sumOfWeights_ = first.sumOfWeights_ - second.sumOfWeights_;
        util::setViewToDifference(this->view.begin(), first.view.cbegin(), second.view.cbegin(), firstIndices.cbegin(),
                                  this->getNumElements());
    }

    template<typename WeightType>
    void SparseDecomposableStatisticVector<WeightType>::clear() {
        uint32 numElements = this->getNumElements();
        typename View<SparseStatistic<float64, WeightType>>::iterator iterator = this->view.begin();

        for (uint32 i = 0; i < numElements; i++) {
            SparseStatistic<float64, WeightType>& sparseStatistic = iterator[i];
            sparseStatistic.gradient = 0;
            sparseStatistic.hessian = 0;
            sparseStatistic.weight = 0;
        }

        sumOfWeights_ = 0;
    }

    template class SparseDecomposableStatisticVector<uint32>;
    template class SparseDecomposableStatisticVector<float32>;
}
