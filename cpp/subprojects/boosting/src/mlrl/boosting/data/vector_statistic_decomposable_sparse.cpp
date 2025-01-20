#include "mlrl/boosting/data/vector_statistic_decomposable_sparse.hpp"

namespace boosting {

    template<typename StatisticType, typename WeightType>
    SparseDecomposableStatisticVector<StatisticType, WeightType>::ConstIterator::ConstIterator(
      typename View<SparseStatistic<StatisticType, WeightType>>::const_iterator iterator, WeightType sumOfWeights)
        : iterator_(iterator), sumOfWeights_(sumOfWeights) {}

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVector<StatisticType, WeightType>::ConstIterator::value_type
      SparseDecomposableStatisticVector<StatisticType, WeightType>::ConstIterator::operator[](uint32 index) const {
        const SparseStatistic<StatisticType, WeightType>& statistic = iterator_[index];
        StatisticType gradient = statistic.gradient;
        StatisticType hessian = statistic.hessian + (sumOfWeights_ - statistic.weight);
        return Statistic<StatisticType>(gradient, hessian);
    }

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVector<StatisticType, WeightType>::ConstIterator::value_type
      SparseDecomposableStatisticVector<StatisticType, WeightType>::ConstIterator::operator*() const {
        const SparseStatistic<StatisticType, WeightType>& statistic = *iterator_;
        StatisticType gradient = statistic.gradient;
        StatisticType hessian = statistic.hessian + (sumOfWeights_ - statistic.weight);
        return Statistic<StatisticType>(gradient, hessian);
    }

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVector<StatisticType, WeightType>::ConstIterator&
      SparseDecomposableStatisticVector<StatisticType, WeightType>::ConstIterator::operator++() {
        ++iterator_;
        return *this;
    }

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVector<StatisticType, WeightType>::ConstIterator&
      SparseDecomposableStatisticVector<StatisticType, WeightType>::ConstIterator::operator++(int n) {
        iterator_++;
        return *this;
    }

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVector<StatisticType, WeightType>::ConstIterator&
      SparseDecomposableStatisticVector<StatisticType, WeightType>::ConstIterator::operator--() {
        --iterator_;
        return *this;
    }

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVector<StatisticType, WeightType>::ConstIterator&
      SparseDecomposableStatisticVector<StatisticType, WeightType>::ConstIterator::operator--(int n) {
        iterator_--;
        return *this;
    }

    template<typename StatisticType, typename WeightType>
    bool SparseDecomposableStatisticVector<StatisticType, WeightType>::ConstIterator::operator!=(
      const ConstIterator& rhs) const {
        return iterator_ != rhs.iterator_;
    }

    template<typename StatisticType, typename WeightType>
    bool SparseDecomposableStatisticVector<StatisticType, WeightType>::ConstIterator::operator==(
      const ConstIterator& rhs) const {
        return iterator_ == rhs.iterator_;
    }

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVector<StatisticType, WeightType>::ConstIterator::difference_type
      SparseDecomposableStatisticVector<StatisticType, WeightType>::ConstIterator::operator-(
        const ConstIterator& rhs) const {
        return iterator_ - rhs.iterator_;
    }

    template<typename StatisticType, typename WeightType>
    static inline void addToSparseDecomposableStatisticVector(
      typename View<SparseStatistic<StatisticType, WeightType>>::iterator statistics,
      typename SparseSetView<Statistic<StatisticType>>::value_const_iterator begin,
      typename SparseSetView<Statistic<StatisticType>>::value_const_iterator end) {
        uint32 numElements = end - begin;

        for (uint32 i = 0; i < numElements; i++) {
            const IndexedValue<Statistic<StatisticType>>& entry = begin[i];
            const Statistic<StatisticType>& statistic = entry.value;
            SparseStatistic<StatisticType, WeightType>& sparseStatistic = statistics[entry.index];
            sparseStatistic.gradient += statistic.gradient;
            sparseStatistic.hessian += statistic.hessian;
            sparseStatistic.weight += 1;
        }
    }

    template<typename StatisticType, typename WeightType>
    static inline void addToSparseDecomposableStatisticVectorWeighted(
      typename View<SparseStatistic<StatisticType, WeightType>>::iterator statistics,
      typename SparseSetView<Statistic<StatisticType>>::value_const_iterator begin,
      typename SparseSetView<Statistic<StatisticType>>::value_const_iterator end, WeightType weight) {
        uint32 numElements = end - begin;

        for (uint32 i = 0; i < numElements; i++) {
            const IndexedValue<Statistic<StatisticType>>& entry = begin[i];
            const Statistic<StatisticType>& statistic = entry.value;
            SparseStatistic<StatisticType, WeightType>& sparseStatistic = statistics[entry.index];
            sparseStatistic.gradient += (statistic.gradient * weight);
            sparseStatistic.hessian += (statistic.hessian * weight);
            sparseStatistic.weight += weight;
        }
    }

    template<typename StatisticType, typename WeightType>
    static inline void removeFromSparseDecomposableStatisticVector(
      typename View<SparseStatistic<StatisticType, WeightType>>::iterator statistics,
      typename SparseSetView<Statistic<StatisticType>>::value_const_iterator begin,
      typename SparseSetView<Statistic<StatisticType>>::value_const_iterator end) {
        uint32 numElements = end - begin;

        for (uint32 i = 0; i < numElements; i++) {
            const IndexedValue<Statistic<StatisticType>>& entry = begin[i];
            const Statistic<StatisticType>& statistic = entry.value;
            SparseStatistic<StatisticType, WeightType>& sparseStatistic = statistics[entry.index];
            sparseStatistic.gradient -= statistic.gradient;
            sparseStatistic.hessian -= statistic.hessian;
            sparseStatistic.weight -= 1;
        }
    }

    template<typename StatisticType, typename WeightType>
    static inline void removeFromSparseDecomposableStatisticVectorWeighted(
      typename View<SparseStatistic<StatisticType, WeightType>>::iterator statistics,
      typename SparseSetView<Statistic<StatisticType>>::value_const_iterator begin,
      typename SparseSetView<Statistic<StatisticType>>::value_const_iterator end, WeightType weight) {
        uint32 numElements = end - begin;

        for (uint32 i = 0; i < numElements; i++) {
            const IndexedValue<Statistic<StatisticType>>& entry = begin[i];
            const Statistic<StatisticType>& statistic = entry.value;
            SparseStatistic<StatisticType, WeightType>& sparseStatistic = statistics[entry.index];
            sparseStatistic.gradient -= (statistic.gradient * weight);
            sparseStatistic.hessian -= (statistic.hessian * weight);
            sparseStatistic.weight -= weight;
        }
    }

    template<typename StatisticType, typename WeightType>
    SparseDecomposableStatisticVector<StatisticType, WeightType>::SparseDecomposableStatisticVector(uint32 numElements,
                                                                                                    bool init)
        : VectorDecorator<AllocatedVector<SparseStatistic<StatisticType, WeightType>>>(
            AllocatedVector<SparseStatistic<StatisticType, WeightType>>(numElements, init)),
          sumOfWeights_(0) {}

    template<typename StatisticType, typename WeightType>
    SparseDecomposableStatisticVector<StatisticType, WeightType>::SparseDecomposableStatisticVector(
      const SparseDecomposableStatisticVector<StatisticType, WeightType>& other)
        : SparseDecomposableStatisticVector(other.getNumElements()) {
        util::copyView(other.view.cbegin(), this->view.begin(), this->getNumElements());
        sumOfWeights_ = other.sumOfWeights_;
    }

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVector<StatisticType, WeightType>::const_iterator
      SparseDecomposableStatisticVector<StatisticType, WeightType>::cbegin() const {
        return ConstIterator(this->view.cbegin(), sumOfWeights_);
    }

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVector<StatisticType, WeightType>::const_iterator
      SparseDecomposableStatisticVector<StatisticType, WeightType>::cend() const {
        return ConstIterator(this->view.cend(), sumOfWeights_);
    }

    template<typename StatisticType, typename WeightType>
    void SparseDecomposableStatisticVector<StatisticType, WeightType>::add(
      const SparseDecomposableStatisticVector& vector) {
        sumOfWeights_ += vector.sumOfWeights_;
        util::addToView(this->view.begin(), vector.view.cbegin(), this->getNumElements());
    }

    template<typename StatisticType, typename WeightType>
    void SparseDecomposableStatisticVector<StatisticType, WeightType>::add(
      const SparseSetView<Statistic<StatisticType>>& view, uint32 row) {
        sumOfWeights_ += 1;
        addToSparseDecomposableStatisticVector<StatisticType, WeightType>(this->view.begin(), view.values_cbegin(row),
                                                                          view.values_cend(row));
    }

    template<typename StatisticType, typename WeightType>
    void SparseDecomposableStatisticVector<StatisticType, WeightType>::add(
      const SparseSetView<Statistic<StatisticType>>& view, uint32 row, WeightType weight) {
        if (!isEqualToZero(weight)) {
            sumOfWeights_ += weight;
            addToSparseDecomposableStatisticVectorWeighted<StatisticType, WeightType>(
              this->view.begin(), view.values_cbegin(row), view.values_cend(row), weight);
        }
    }

    template<typename StatisticType, typename WeightType>
    void SparseDecomposableStatisticVector<StatisticType, WeightType>::remove(
      const SparseSetView<Statistic<StatisticType>>& view, uint32 row) {
        sumOfWeights_ -= 1;
        removeFromSparseDecomposableStatisticVector<StatisticType, WeightType>(
          this->view.begin(), view.values_cbegin(row), view.values_cend(row));
    }

    template<typename StatisticType, typename WeightType>
    void SparseDecomposableStatisticVector<StatisticType, WeightType>::remove(
      const SparseSetView<Statistic<StatisticType>>& view, uint32 row, WeightType weight) {
        if (!isEqualToZero(weight)) {
            sumOfWeights_ -= weight;
            removeFromSparseDecomposableStatisticVectorWeighted<StatisticType, WeightType>(
              this->view.begin(), view.values_cbegin(row), view.values_cend(row), weight);
        }
    }

    template<typename StatisticType, typename WeightType>
    void SparseDecomposableStatisticVector<StatisticType, WeightType>::addToSubset(
      const SparseSetView<Statistic<StatisticType>>& view, uint32 row, const CompleteIndexVector& indices) {
        sumOfWeights_ += 1;
        addToSparseDecomposableStatisticVector<StatisticType, WeightType>(this->view.begin(), view.values_cbegin(row),
                                                                          view.values_cend(row));
    }

    template<typename StatisticType, typename WeightType>
    void SparseDecomposableStatisticVector<StatisticType, WeightType>::addToSubset(
      const SparseSetView<Statistic<StatisticType>>& view, uint32 row, const PartialIndexVector& indices) {
        sumOfWeights_ += 1;
        typename SparseSetView<Statistic<StatisticType>>::const_row viewRow = view[row];
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        uint32 numElements = indices.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];
            const IndexedValue<Statistic<StatisticType>>* entry = viewRow[index];

            if (entry) {
                const Statistic<StatisticType>& statistic = entry->value;
                SparseStatistic<StatisticType, WeightType>& sparseStatistic = this->view.begin()[i];
                sparseStatistic.gradient += (statistic.gradient);
                sparseStatistic.hessian += (statistic.hessian);
                sparseStatistic.weight += 1;
            }
        }
    }

    template<typename StatisticType, typename WeightType>
    void SparseDecomposableStatisticVector<StatisticType, WeightType>::addToSubset(
      const SparseSetView<Statistic<StatisticType>>& view, uint32 row, const CompleteIndexVector& indices,
      WeightType weight) {
        if (!isEqualToZero(weight)) {
            sumOfWeights_ += weight;
            addToSparseDecomposableStatisticVectorWeighted<StatisticType, WeightType>(
              this->view.begin(), view.values_cbegin(row), view.values_cend(row), weight);
        }
    }

    template<typename StatisticType, typename WeightType>
    void SparseDecomposableStatisticVector<StatisticType, WeightType>::addToSubset(
      const SparseSetView<Statistic<StatisticType>>& view, uint32 row, const PartialIndexVector& indices,
      WeightType weight) {
        if (!isEqualToZero(weight)) {
            sumOfWeights_ += weight;
            typename SparseSetView<Statistic<StatisticType>>::const_row viewRow = view[row];
            PartialIndexVector::const_iterator indexIterator = indices.cbegin();
            uint32 numElements = indices.getNumElements();

            for (uint32 i = 0; i < numElements; i++) {
                uint32 index = indexIterator[i];
                const IndexedValue<Statistic<StatisticType>>* entry = viewRow[index];

                if (entry) {
                    const Statistic<StatisticType>& statistic = entry->value;
                    SparseStatistic<StatisticType, WeightType>& sparseStatistic = this->view.begin()[i];
                    sparseStatistic.gradient += (statistic.gradient * weight);
                    sparseStatistic.hessian += (statistic.hessian * weight);
                    sparseStatistic.weight += weight;
                }
            }
        }
    }

    template<typename StatisticType, typename WeightType>
    void SparseDecomposableStatisticVector<StatisticType, WeightType>::difference(
      const SparseDecomposableStatisticVector<StatisticType, WeightType>& first,
      const CompleteIndexVector& firstIndices,
      const SparseDecomposableStatisticVector<StatisticType, WeightType>& second) {
        sumOfWeights_ = first.sumOfWeights_ - second.sumOfWeights_;
        util::setViewToDifference(this->view.begin(), first.view.cbegin(), second.view.cbegin(),
                                  this->getNumElements());
    }

    template<typename StatisticType, typename WeightType>
    void SparseDecomposableStatisticVector<StatisticType, WeightType>::difference(
      const SparseDecomposableStatisticVector<StatisticType, WeightType>& first, const PartialIndexVector& firstIndices,
      const SparseDecomposableStatisticVector<StatisticType, WeightType>& second) {
        sumOfWeights_ = first.sumOfWeights_ - second.sumOfWeights_;
        util::setViewToDifference(this->view.begin(), first.view.cbegin(), second.view.cbegin(), firstIndices.cbegin(),
                                  this->getNumElements());
    }

    template<typename StatisticType, typename WeightType>
    void SparseDecomposableStatisticVector<StatisticType, WeightType>::clear() {
        uint32 numElements = this->getNumElements();
        typename View<SparseStatistic<StatisticType, WeightType>>::iterator iterator = this->view.begin();

        for (uint32 i = 0; i < numElements; i++) {
            SparseStatistic<StatisticType, WeightType>& sparseStatistic = iterator[i];
            sparseStatistic.gradient = 0;
            sparseStatistic.hessian = 0;
            sparseStatistic.weight = 0;
        }

        sumOfWeights_ = 0;
    }

    template class SparseDecomposableStatisticVector<float32, uint32>;
    template class SparseDecomposableStatisticVector<float32, float32>;
    template class SparseDecomposableStatisticVector<float64, uint32>;
    template class SparseDecomposableStatisticVector<float64, float32>;
}
