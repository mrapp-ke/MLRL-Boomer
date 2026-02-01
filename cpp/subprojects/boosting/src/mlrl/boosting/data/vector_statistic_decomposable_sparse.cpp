#include "mlrl/boosting/data/vector_statistic_decomposable_sparse.hpp"

#include "mlrl/common/util/array_operations.hpp"
#include "mlrl/common/util/xsimd.hpp"

namespace boosting {

    template<typename StatisticType, typename WeightType>
    SparseDecomposableStatisticVectorView<StatisticType, WeightType>::GradientConstIterator::GradientConstIterator(
      typename View<SparseStatistic<StatisticType, WeightType>>::const_iterator iterator, WeightType sumOfWeights)
        : iterator_(iterator), sumOfWeights_(sumOfWeights) {}

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVectorView<StatisticType, WeightType>::GradientConstIterator::value_type
      SparseDecomposableStatisticVectorView<StatisticType, WeightType>::GradientConstIterator::operator[](
        uint32 index) const {
        return iterator_[index].gradient;
    }

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVectorView<StatisticType, WeightType>::GradientConstIterator::value_type
      SparseDecomposableStatisticVectorView<StatisticType, WeightType>::GradientConstIterator::operator*() const {
        return (*iterator_).gradient;
    }

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVectorView<StatisticType, WeightType>::GradientConstIterator&
      SparseDecomposableStatisticVectorView<StatisticType, WeightType>::GradientConstIterator::operator++() {
        ++iterator_;
        return *this;
    }

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVectorView<StatisticType, WeightType>::GradientConstIterator&
      SparseDecomposableStatisticVectorView<StatisticType, WeightType>::GradientConstIterator::operator++(int n) {
        iterator_++;
        return *this;
    }

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVectorView<StatisticType, WeightType>::GradientConstIterator&
      SparseDecomposableStatisticVectorView<StatisticType, WeightType>::GradientConstIterator::operator--() {
        --iterator_;
        return *this;
    }

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVectorView<StatisticType, WeightType>::GradientConstIterator&
      SparseDecomposableStatisticVectorView<StatisticType, WeightType>::GradientConstIterator::operator--(int n) {
        iterator_--;
        return *this;
    }

    template<typename StatisticType, typename WeightType>
    bool SparseDecomposableStatisticVectorView<StatisticType, WeightType>::GradientConstIterator::operator!=(
      const GradientConstIterator& rhs) const {
        return iterator_ != rhs.iterator_;
    }

    template<typename StatisticType, typename WeightType>
    bool SparseDecomposableStatisticVectorView<StatisticType, WeightType>::GradientConstIterator::operator==(
      const GradientConstIterator& rhs) const {
        return iterator_ == rhs.iterator_;
    }

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVectorView<StatisticType, WeightType>::GradientConstIterator::difference_type
      SparseDecomposableStatisticVectorView<StatisticType, WeightType>::GradientConstIterator::operator-(
        const GradientConstIterator& rhs) const {
        return iterator_ - rhs.iterator_;
    }

    template<typename StatisticType, typename WeightType>
    SparseDecomposableStatisticVectorView<StatisticType, WeightType>::HessianConstIterator::HessianConstIterator(
      typename View<SparseStatistic<StatisticType, WeightType>>::const_iterator iterator, WeightType sumOfWeights)
        : iterator_(iterator), sumOfWeights_(sumOfWeights) {}

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVectorView<StatisticType, WeightType>::HessianConstIterator::value_type
      SparseDecomposableStatisticVectorView<StatisticType, WeightType>::HessianConstIterator::operator[](
        uint32 index) const {
        const SparseStatistic<StatisticType, WeightType>& statistic = iterator_[index];
        return statistic.hessian + (sumOfWeights_ - statistic.weight);
    }

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVectorView<StatisticType, WeightType>::HessianConstIterator::value_type
      SparseDecomposableStatisticVectorView<StatisticType, WeightType>::HessianConstIterator::operator*() const {
        const SparseStatistic<StatisticType, WeightType>& statistic = *iterator_;
        return statistic.hessian + (sumOfWeights_ - statistic.weight);
    }

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVectorView<StatisticType, WeightType>::HessianConstIterator&
      SparseDecomposableStatisticVectorView<StatisticType, WeightType>::HessianConstIterator::operator++() {
        ++iterator_;
        return *this;
    }

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVectorView<StatisticType, WeightType>::HessianConstIterator&
      SparseDecomposableStatisticVectorView<StatisticType, WeightType>::HessianConstIterator::operator++(int n) {
        iterator_++;
        return *this;
    }

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVectorView<StatisticType, WeightType>::HessianConstIterator&
      SparseDecomposableStatisticVectorView<StatisticType, WeightType>::HessianConstIterator::operator--() {
        --iterator_;
        return *this;
    }

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVectorView<StatisticType, WeightType>::HessianConstIterator&
      SparseDecomposableStatisticVectorView<StatisticType, WeightType>::HessianConstIterator::operator--(int n) {
        iterator_--;
        return *this;
    }

    template<typename StatisticType, typename WeightType>
    bool SparseDecomposableStatisticVectorView<StatisticType, WeightType>::HessianConstIterator::operator!=(
      const HessianConstIterator& rhs) const {
        return iterator_ != rhs.iterator_;
    }

    template<typename StatisticType, typename WeightType>
    bool SparseDecomposableStatisticVectorView<StatisticType, WeightType>::HessianConstIterator::operator==(
      const HessianConstIterator& rhs) const {
        return iterator_ == rhs.iterator_;
    }

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVectorView<StatisticType, WeightType>::HessianConstIterator::difference_type
      SparseDecomposableStatisticVectorView<StatisticType, WeightType>::HessianConstIterator::operator-(
        const HessianConstIterator& rhs) const {
        return iterator_ - rhs.iterator_;
    }

    template<typename StatisticType, typename WeightType>
    SparseDecomposableStatisticVectorView<StatisticType, WeightType>::SparseDecomposableStatisticVectorView(
      uint32 numElements, bool init)
        : AllocatedVector<SparseStatistic<StatisticType, WeightType>>(numElements, init), sumOfWeights(0) {}

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVectorView<StatisticType, WeightType>::gradient_const_iterator
      SparseDecomposableStatisticVectorView<StatisticType, WeightType>::gradients_cbegin() const {
        return GradientConstIterator(this->cbegin(), sumOfWeights);
    }

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVectorView<StatisticType, WeightType>::gradient_const_iterator
      SparseDecomposableStatisticVectorView<StatisticType, WeightType>::gradients_cend() const {
        return GradientConstIterator(this->cend(), sumOfWeights);
    }

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVectorView<StatisticType, WeightType>::hessian_const_iterator
      SparseDecomposableStatisticVectorView<StatisticType, WeightType>::hessians_cbegin() const {
        return HessianConstIterator(this->cbegin(), sumOfWeights);
    }

    template<typename StatisticType, typename WeightType>
    typename SparseDecomposableStatisticVectorView<StatisticType, WeightType>::hessian_const_iterator
      SparseDecomposableStatisticVectorView<StatisticType, WeightType>::hessians_cend() const {
        return HessianConstIterator(this->cend(), sumOfWeights);
    }

    template<typename StatisticType, typename WeightType>
    const uint32 SparseDecomposableStatisticVectorView<StatisticType, WeightType>::getNumElements() const {
        return this->numElements;
    }

    template class SparseDecomposableStatisticVectorView<float32, uint32>;
    template class SparseDecomposableStatisticVectorView<float32, float32>;
    template class SparseDecomposableStatisticVectorView<float64, uint32>;
    template class SparseDecomposableStatisticVectorView<float64, float32>;

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

    template<typename StatisticType, typename WeightType, typename ArrayOperations>
    SparseDecomposableStatisticVector<StatisticType, WeightType, ArrayOperations>::SparseDecomposableStatisticVector(
      uint32 numElements, bool init)
        : VectorDecorator<SparseDecomposableStatisticVectorView<StatisticType, WeightType>>(
            SparseDecomposableStatisticVectorView<StatisticType, WeightType>(numElements, init)) {}

    template<typename StatisticType, typename WeightType, typename ArrayOperations>
    SparseDecomposableStatisticVector<StatisticType, WeightType, ArrayOperations>::SparseDecomposableStatisticVector(
      const SparseDecomposableStatisticVector<StatisticType, WeightType, ArrayOperations>& other)
        : SparseDecomposableStatisticVector<StatisticType, WeightType, ArrayOperations>(other.getNumElements()) {
        ArrayOperations::copy(other.view.cbegin(), this->view.begin(), this->getNumElements());
        this->view.sumOfWeights = other.view.sumOfWeights;
    }

    template<typename StatisticType, typename WeightType, typename ArrayOperations>
    void SparseDecomposableStatisticVector<StatisticType, WeightType, ArrayOperations>::add(
      const SparseDecomposableStatisticVectorView<StatisticType, WeightType>& vector) {
        this->view.sumOfWeights += vector.sumOfWeights;
        ArrayOperations::add(this->view.begin(), vector.cbegin(), this->getNumElements());
    }

    template<typename StatisticType, typename WeightType, typename ArrayOperations>
    void SparseDecomposableStatisticVector<StatisticType, WeightType, ArrayOperations>::add(
      const SparseSetView<Statistic<StatisticType>>& view, uint32 row) {
        this->view.sumOfWeights += 1;
        addToSparseDecomposableStatisticVector<StatisticType, WeightType>(this->view.begin(), view.values_cbegin(row),
                                                                          view.values_cend(row));
    }

    template<typename StatisticType, typename WeightType, typename ArrayOperations>
    void SparseDecomposableStatisticVector<StatisticType, WeightType, ArrayOperations>::add(
      const SparseSetView<Statistic<StatisticType>>& view, uint32 row, WeightType weight) {
        if (!isEqualToZero(weight)) {
            this->view.sumOfWeights += weight;
            addToSparseDecomposableStatisticVectorWeighted<StatisticType, WeightType>(
              this->view.begin(), view.values_cbegin(row), view.values_cend(row), weight);
        }
    }

    template<typename StatisticType, typename WeightType, typename ArrayOperations>
    void SparseDecomposableStatisticVector<StatisticType, WeightType, ArrayOperations>::remove(
      const SparseSetView<Statistic<StatisticType>>& view, uint32 row) {
        this->view.sumOfWeights -= 1;
        removeFromSparseDecomposableStatisticVector<StatisticType, WeightType>(
          this->view.begin(), view.values_cbegin(row), view.values_cend(row));
    }

    template<typename StatisticType, typename WeightType, typename ArrayOperations>
    void SparseDecomposableStatisticVector<StatisticType, WeightType, ArrayOperations>::remove(
      const SparseSetView<Statistic<StatisticType>>& view, uint32 row, WeightType weight) {
        if (!isEqualToZero(weight)) {
            this->view.sumOfWeights -= weight;
            removeFromSparseDecomposableStatisticVectorWeighted<StatisticType, WeightType>(
              this->view.begin(), view.values_cbegin(row), view.values_cend(row), weight);
        }
    }

    template<typename StatisticType, typename WeightType, typename ArrayOperations>
    void SparseDecomposableStatisticVector<StatisticType, WeightType, ArrayOperations>::addToSubset(
      const SparseSetView<Statistic<StatisticType>>& view, uint32 row, const CompleteIndexVector& indices) {
        this->view.sumOfWeights += 1;
        addToSparseDecomposableStatisticVector<StatisticType, WeightType>(this->view.begin(), view.values_cbegin(row),
                                                                          view.values_cend(row));
    }

    template<typename StatisticType, typename WeightType, typename ArrayOperations>
    void SparseDecomposableStatisticVector<StatisticType, WeightType, ArrayOperations>::addToSubset(
      const SparseSetView<Statistic<StatisticType>>& view, uint32 row, const PartialIndexVector& indices) {
        this->view.sumOfWeights += 1;
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

    template<typename StatisticType, typename WeightType, typename ArrayOperations>
    void SparseDecomposableStatisticVector<StatisticType, WeightType, ArrayOperations>::addToSubset(
      const SparseSetView<Statistic<StatisticType>>& view, uint32 row, const CompleteIndexVector& indices,
      WeightType weight) {
        if (!isEqualToZero(weight)) {
            this->view.sumOfWeights += weight;
            addToSparseDecomposableStatisticVectorWeighted<StatisticType, WeightType>(
              this->view.begin(), view.values_cbegin(row), view.values_cend(row), weight);
        }
    }

    template<typename StatisticType, typename WeightType, typename ArrayOperations>
    void SparseDecomposableStatisticVector<StatisticType, WeightType, ArrayOperations>::addToSubset(
      const SparseSetView<Statistic<StatisticType>>& view, uint32 row, const PartialIndexVector& indices,
      WeightType weight) {
        if (!isEqualToZero(weight)) {
            this->view.sumOfWeights += weight;
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

    template<typename StatisticType, typename WeightType, typename ArrayOperations>
    void SparseDecomposableStatisticVector<StatisticType, WeightType, ArrayOperations>::difference(
      const SparseDecomposableStatisticVectorView<StatisticType, WeightType>& first,
      const CompleteIndexVector& firstIndices,
      const SparseDecomposableStatisticVectorView<StatisticType, WeightType>& second) {
        this->view.sumOfWeights = first.sumOfWeights - second.sumOfWeights;
        ArrayOperations::difference(this->view.begin(), first.cbegin(), second.cbegin(), this->getNumElements());
    }

    template<typename StatisticType, typename WeightType, typename ArrayOperations>
    void SparseDecomposableStatisticVector<StatisticType, WeightType, ArrayOperations>::difference(
      const SparseDecomposableStatisticVectorView<StatisticType, WeightType>& first,
      const PartialIndexVector& firstIndices,
      const SparseDecomposableStatisticVectorView<StatisticType, WeightType>& second) {
        this->view.sumOfWeights = first.sumOfWeights - second.sumOfWeights;
        ArrayOperations::difference(this->view.begin(), first.cbegin(), second.cbegin(), firstIndices.cbegin(),
                                    this->getNumElements());
    }

    template<typename StatisticType, typename WeightType, typename ArrayOperations>
    void SparseDecomposableStatisticVector<StatisticType, WeightType, ArrayOperations>::clear() {
        uint32 numElements = this->getNumElements();
        typename View<SparseStatistic<StatisticType, WeightType>>::iterator iterator = this->view.begin();

        for (uint32 i = 0; i < numElements; i++) {
            SparseStatistic<StatisticType, WeightType>& sparseStatistic = iterator[i];
            sparseStatistic.gradient = 0;
            sparseStatistic.hessian = 0;
            sparseStatistic.weight = 0;
        }

        this->view.sumOfWeights = 0;
    }

    template class SparseDecomposableStatisticVector<float32, uint32, SequentialArrayOperations>;
    template class SparseDecomposableStatisticVector<float32, float32, SequentialArrayOperations>;
    template class SparseDecomposableStatisticVector<float64, uint32, SequentialArrayOperations>;
    template class SparseDecomposableStatisticVector<float64, float32, SequentialArrayOperations>;

#if SIMD_SUPPORT_ENABLED
    template class SparseDecomposableStatisticVector<float32, uint32, SimdArrayOperations>;
    template class SparseDecomposableStatisticVector<float32, float32, SimdArrayOperations>;
    template class SparseDecomposableStatisticVector<float64, uint32, SimdArrayOperations>;
    template class SparseDecomposableStatisticVector<float64, float32, SimdArrayOperations>;
#endif
}
