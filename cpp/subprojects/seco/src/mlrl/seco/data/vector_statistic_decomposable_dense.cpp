#include "mlrl/seco/data/vector_statistic_decomposable_dense.hpp"

#include "mlrl/common/iterator/iterator_forward_sparse_binary.hpp"
#include "mlrl/common/math/vector_math.hpp"
#include "mlrl/common/simd/memory.hpp"
#include "mlrl/common/simd/vector_math.hpp"

namespace seco {

    template<typename T, typename Constant>
    static inline void addConstantToSubset(T* array, Constant constant, const uint32* indices, uint32 numIndices,
                                           const PartialIndexVector& indexVector) {
        auto indexVectorIterator = indexVector.cbegin();
        auto indexVectorEnd = indexVector.cend();
        uint32 n = 0;

        for (uint32 i = 0; i < numIndices; i++) {
            uint32 index = indices[i];

            while (indexVectorIterator != indexVectorEnd && *indexVectorIterator < index) {
                indexVectorIterator++;
                n++;
            }

            if (indexVectorIterator != indexVectorEnd && *indexVectorIterator == index) {
                array[n] += constant;
            }
        }
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::DenseDecomposableStatisticVector(
      uint32 numElements, bool init)
        : ClearableViewDecorator<ViewDecorator<DenseDecomposableStatisticVectorAllocator<
            DenseDecomposableStatisticVectorView<StatisticType>, MemoryAllocator>>>(
            DenseDecomposableStatisticVectorAllocator<DenseDecomposableStatisticVectorView<StatisticType>,
                                                      MemoryAllocator>(numElements, init)) {}

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::DenseDecomposableStatisticVector(
      const DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>& other)
        : DenseDecomposableStatisticVector(other.getNumElements()) {
        VectorMath::copy(other.view.cbegin(), this->view.begin(), this->view.numElements);
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::add(
      const DenseDecomposableStatisticVectorView<StatisticType>& other) {
        VectorMath::add(this->view.begin(), other.cbegin(), this->view.numElements);
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    typename View<StatisticType>::const_iterator
      DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::correct_counts_cbegin() const {
        return this->view.correct_counts_cbegin();
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    typename View<StatisticType>::const_iterator
      DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::correct_counts_cend() const {
        return this->view.correct_counts_cend();
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    typename View<StatisticType>::iterator
      DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::correct_counts_begin() {
        return this->view.correct_counts_begin();
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    typename View<StatisticType>::iterator
      DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::correct_counts_end() {
        return this->view.correct_counts_end();
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    typename View<StatisticType>::const_iterator
      DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::incorrect_counts_cbegin() const {
        return this->view.incorrect_counts_cbegin();
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    typename View<StatisticType>::const_iterator
      DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::incorrect_counts_cend() const {
        return this->view.incorrect_counts_cend();
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    typename View<StatisticType>::iterator
      DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::incorrect_counts_begin() {
        return this->view.incorrect_counts_begin();
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    typename View<StatisticType>::iterator
      DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::incorrect_counts_end() {
        return this->view.incorrect_counts_end();
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    uint32 DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::getNumElements() const {
        return this->view.numCounts;
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::add(
      const SparseDecomposableStatisticView& view, uint32 row, StatisticType weight) {
        VectorMath::addConstantToSubset(this->correct_counts_begin(), weight, view.correct_indices_cbegin(row),
                                        view.correct_indices_cend(row) - view.correct_indices_cbegin(row));
        VectorMath::addConstantToSubset(this->incorrect_counts_begin(), weight, view.incorrect_indices_cbegin(row),
                                        view.incorrect_indices_cend(row) - view.incorrect_indices_cbegin(row));
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::remove(
      const SparseDecomposableStatisticView& view, uint32 row, StatisticType weight) {
        VectorMath::subtractConstantFromSubset(this->correct_counts_begin(), weight, view.correct_indices_cbegin(row),
                                               view.correct_indices_cend(row) - view.correct_indices_cbegin(row));
        VectorMath::subtractConstantFromSubset(this->incorrect_counts_begin(), weight,
                                               view.incorrect_indices_cbegin(row),
                                               view.incorrect_indices_cend(row) - view.incorrect_indices_cbegin(row));
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::addToSubset(
      const SparseDecomposableStatisticView& view, uint32 row, const CompleteIndexVector& indices,
      StatisticType weight) {
        VectorMath::addConstantToSubset(this->correct_counts_begin(), weight, view.correct_indices_cbegin(row),
                                        view.correct_indices_cend(row) - view.correct_indices_cbegin(row));
        VectorMath::addConstantToSubset(this->incorrect_counts_begin(), weight, view.incorrect_indices_cbegin(row),
                                        view.incorrect_indices_cend(row) - view.incorrect_indices_cbegin(row));
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::addToSubset(
      const SparseDecomposableStatisticView& view, uint32 row, const PartialIndexVector& indices,
      StatisticType weight) {
        addConstantToSubset(this->correct_counts_begin(), weight, view.correct_indices_cbegin(row),
                            view.correct_indices_cend(row) - view.correct_indices_cbegin(row), indices);
        addConstantToSubset(this->incorrect_counts_begin(), weight, view.incorrect_indices_cbegin(row),
                            view.incorrect_indices_cend(row) - view.incorrect_indices_cbegin(row), indices);
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::difference(
      const DenseDecomposableStatisticVectorView<StatisticType>& first, const CompleteIndexVector& firstIndices,
      const DenseDecomposableStatisticVectorView<StatisticType>& second) {
        VectorMath::difference(this->view.begin(), first.cbegin(), second.cbegin(), this->view.numElements);
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::difference(
      const DenseDecomposableStatisticVectorView<StatisticType>& first, const PartialIndexVector& firstIndices,
      const DenseDecomposableStatisticVectorView<StatisticType>& second) {
        uint32 numElements = this->getNumElements();
        auto indexIterator = firstIndices.cbegin();
        VectorMath::difference(this->correct_counts_begin(), first.correct_counts_cbegin(),
                               second.correct_counts_cbegin(), indexIterator, numElements);
        VectorMath::difference(this->incorrect_counts_begin(), first.incorrect_counts_cbegin(),
                               second.incorrect_counts_cbegin(), indexIterator, numElements);
    }

    template class DenseDecomposableStatisticVector<uint32, DefaultMemoryAllocator, SequentialVectorMath>;
    template class DenseDecomposableStatisticVector<float32, DefaultMemoryAllocator, SequentialVectorMath>;

#if SIMD_SUPPORT_ENABLED
    template class DenseDecomposableStatisticVector<uint32, SimdMemoryAllocator, SimdVectorMath>;
    template class DenseDecomposableStatisticVector<float32, SimdMemoryAllocator, SimdVectorMath>;
#endif
}
