#include "mlrl/boosting/data/vector_statistic_decomposable_sparse.hpp"

#include "mlrl/common/math/vector_math.hpp"

namespace boosting {

    template<typename StatisticType, typename WeightType>
    static inline void addToSparseDecomposableStatisticVector(
      typename View<SparseStatistic<StatisticType, WeightType>>::iterator statistics,
      typename SparseDecomposableStatisticView<StatisticType>::value_const_iterator begin,
      typename SparseDecomposableStatisticView<StatisticType>::value_const_iterator end) {
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
      typename SparseDecomposableStatisticView<StatisticType>::value_const_iterator begin,
      typename SparseDecomposableStatisticView<StatisticType>::value_const_iterator end, WeightType weight) {
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
      typename SparseDecomposableStatisticView<StatisticType>::value_const_iterator begin,
      typename SparseDecomposableStatisticView<StatisticType>::value_const_iterator end) {
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
      typename SparseDecomposableStatisticView<StatisticType>::value_const_iterator begin,
      typename SparseDecomposableStatisticView<StatisticType>::value_const_iterator end, WeightType weight) {
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
        : VectorDecorator<Allocator<SparseDecomposableStatisticVectorView<StatisticType, WeightType>>>(
            Allocator<SparseDecomposableStatisticVectorView<StatisticType, WeightType>>(numElements, init)) {}

    template<typename StatisticType, typename WeightType>
    SparseDecomposableStatisticVector<StatisticType, WeightType>::SparseDecomposableStatisticVector(
      const SparseDecomposableStatisticVector<StatisticType, WeightType>& other)
        : SparseDecomposableStatisticVector<StatisticType, WeightType>(other.getNumElements()) {
        SequentialVectorMath::copy(other.view.cbegin(), this->view.begin(), this->getNumElements());
        this->view.sumOfWeights = other.view.sumOfWeights;
    }

    template<typename StatisticType, typename WeightType>
    void SparseDecomposableStatisticVector<StatisticType, WeightType>::add(
      const SparseDecomposableStatisticVectorView<StatisticType, WeightType>& vector) {
        this->view.sumOfWeights += vector.sumOfWeights;
        SequentialVectorMath::add(this->view.begin(), vector.cbegin(), this->getNumElements());
    }

    template<typename StatisticType, typename WeightType>
    void SparseDecomposableStatisticVector<StatisticType, WeightType>::add(
      const SparseDecomposableStatisticView<StatisticType>& view, uint32 row) {
        this->view.sumOfWeights += 1;
        addToSparseDecomposableStatisticVector<StatisticType, WeightType>(this->view.begin(), view.values_cbegin(row),
                                                                          view.values_cend(row));
    }

    template<typename StatisticType, typename WeightType>
    void SparseDecomposableStatisticVector<StatisticType, WeightType>::add(
      const SparseDecomposableStatisticView<StatisticType>& view, uint32 row, WeightType weight) {
        if (!isEqualToZero(weight)) {
            this->view.sumOfWeights += weight;
            addToSparseDecomposableStatisticVectorWeighted<StatisticType, WeightType>(
              this->view.begin(), view.values_cbegin(row), view.values_cend(row), weight);
        }
    }

    template<typename StatisticType, typename WeightType>
    void SparseDecomposableStatisticVector<StatisticType, WeightType>::remove(
      const SparseDecomposableStatisticView<StatisticType>& view, uint32 row) {
        this->view.sumOfWeights -= 1;
        removeFromSparseDecomposableStatisticVector<StatisticType, WeightType>(
          this->view.begin(), view.values_cbegin(row), view.values_cend(row));
    }

    template<typename StatisticType, typename WeightType>
    void SparseDecomposableStatisticVector<StatisticType, WeightType>::remove(
      const SparseDecomposableStatisticView<StatisticType>& view, uint32 row, WeightType weight) {
        if (!isEqualToZero(weight)) {
            this->view.sumOfWeights -= weight;
            removeFromSparseDecomposableStatisticVectorWeighted<StatisticType, WeightType>(
              this->view.begin(), view.values_cbegin(row), view.values_cend(row), weight);
        }
    }

    template<typename StatisticType, typename WeightType>
    void SparseDecomposableStatisticVector<StatisticType, WeightType>::addToSubset(
      const SparseDecomposableStatisticView<StatisticType>& view, uint32 row, const CompleteIndexVector& indices) {
        this->view.sumOfWeights += 1;
        addToSparseDecomposableStatisticVector<StatisticType, WeightType>(this->view.begin(), view.values_cbegin(row),
                                                                          view.values_cend(row));
    }

    template<typename StatisticType, typename WeightType>
    void SparseDecomposableStatisticVector<StatisticType, WeightType>::addToSubset(
      const SparseDecomposableStatisticView<StatisticType>& view, uint32 row, const PartialIndexVector& indices) {
        this->view.sumOfWeights += 1;
        const auto viewRow = view[row];
        auto indexIterator = indices.cbegin();
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
      const SparseDecomposableStatisticView<StatisticType>& view, uint32 row, const CompleteIndexVector& indices,
      WeightType weight) {
        if (!isEqualToZero(weight)) {
            this->view.sumOfWeights += weight;
            addToSparseDecomposableStatisticVectorWeighted<StatisticType, WeightType>(
              this->view.begin(), view.values_cbegin(row), view.values_cend(row), weight);
        }
    }

    template<typename StatisticType, typename WeightType>
    void SparseDecomposableStatisticVector<StatisticType, WeightType>::addToSubset(
      const SparseDecomposableStatisticView<StatisticType>& view, uint32 row, const PartialIndexVector& indices,
      WeightType weight) {
        if (!isEqualToZero(weight)) {
            this->view.sumOfWeights += weight;
            const auto viewRow = view[row];
            auto indexIterator = indices.cbegin();
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
      const SparseDecomposableStatisticVectorView<StatisticType, WeightType>& first,
      const CompleteIndexVector& firstIndices,
      const SparseDecomposableStatisticVectorView<StatisticType, WeightType>& second) {
        this->view.sumOfWeights = first.sumOfWeights - second.sumOfWeights;
        SequentialVectorMath::difference(this->view.begin(), first.cbegin(), second.cbegin(), this->getNumElements());
    }

    template<typename StatisticType, typename WeightType>
    void SparseDecomposableStatisticVector<StatisticType, WeightType>::difference(
      const SparseDecomposableStatisticVectorView<StatisticType, WeightType>& first,
      const PartialIndexVector& firstIndices,
      const SparseDecomposableStatisticVectorView<StatisticType, WeightType>& second) {
        this->view.sumOfWeights = first.sumOfWeights - second.sumOfWeights;
        SequentialVectorMath::difference(this->view.begin(), first.cbegin(), second.cbegin(), firstIndices.cbegin(),
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

        this->view.sumOfWeights = 0;
    }

    template class SparseDecomposableStatisticVector<float32, uint32>;
    template class SparseDecomposableStatisticVector<float32, float32>;
    template class SparseDecomposableStatisticVector<float64, uint32>;
    template class SparseDecomposableStatisticVector<float64, float32>;
}
