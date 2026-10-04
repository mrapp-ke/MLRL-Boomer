#include "mlrl/boosting/data/vector_statistic_decomposable_dense.hpp"

#include "mlrl/common/math/vector_math.hpp"
#include "mlrl/common/simd/memory.hpp"
#include "mlrl/common/simd/vector_math.hpp"

namespace boosting {

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::DenseDecomposableStatisticVector(
      uint32 numElements, bool init)
        : ClearableViewDecorator<ViewDecorator<
            DenseStatisticVectorAllocator<DenseDecomposableStatisticVectorView<StatisticType>, MemoryAllocator>>>(
            DenseStatisticVectorAllocator<DenseDecomposableStatisticVectorView<StatisticType>, MemoryAllocator>(
              numElements, numElements, init)) {}

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::DenseDecomposableStatisticVector(
      const DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>& other)
        : DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>(other.getNumElements()) {
        VectorMath::copy(other.view.gradients_cbegin(), this->view.gradients_begin(), other.view.getNumGradients());
        VectorMath::copy(other.view.hessians_cbegin(), this->view.hessians_begin(), other.view.getNumHessians());
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    uint32 DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::getNumElements() const {
        return this->view.getNumGradients();
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::add(
      const DenseDecomposableStatisticVectorView<StatisticType>& vector) {
        VectorMath::add(this->view.gradients_begin(), vector.gradients_cbegin(), this->view.getNumGradients());
        VectorMath::add(this->view.hessians_begin(), vector.hessians_cbegin(), this->view.getNumHessians());
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::add(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row) {
        VectorMath::add(this->view.gradients_begin(), view.gradients_cbegin(row), this->view.getNumGradients());
        VectorMath::add(this->view.hessians_begin(), view.hessians_cbegin(row), this->view.getNumHessians());
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::add(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, StatisticType weight) {
        VectorMath::addWeighted(this->view.gradients_begin(), view.gradients_cbegin(row), this->view.getNumGradients(),
                                weight);
        VectorMath::addWeighted(this->view.hessians_begin(), view.hessians_cbegin(row), this->view.getNumHessians(),
                                weight);
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::remove(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row) {
        VectorMath::subtract(this->view.gradients_begin(), view.gradients_cbegin(row), this->view.getNumGradients());
        VectorMath::subtract(this->view.hessians_begin(), view.hessians_cbegin(row), this->view.getNumHessians());
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::remove(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, StatisticType weight) {
        VectorMath::subtractWeighted(this->view.gradients_begin(), view.gradients_cbegin(row),
                                     this->view.getNumGradients(), weight);
        VectorMath::subtractWeighted(this->view.hessians_begin(), view.hessians_cbegin(row),
                                     this->view.getNumHessians(), weight);
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::addToSubset(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, const CompleteIndexVector& indices) {
        VectorMath::add(this->view.gradients_begin(), view.gradients_cbegin(row), this->view.getNumGradients());
        VectorMath::add(this->view.hessians_begin(), view.hessians_cbegin(row), this->view.getNumHessians());
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::addToSubset(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, const PartialIndexVector& indices) {
        auto indexIterator = indices.cbegin();
        VectorMath::addToSubset(this->view.gradients_begin(), view.gradients_cbegin(row), indexIterator,
                                this->view.getNumGradients());
        VectorMath::addToSubset(this->view.hessians_begin(), view.hessians_cbegin(row), indexIterator,
                                this->view.getNumHessians());
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::addToSubset(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, const CompleteIndexVector& indices,
      StatisticType weight) {
        VectorMath::addWeighted(this->view.gradients_begin(), view.gradients_cbegin(row), this->view.getNumGradients(),
                                weight);
        VectorMath::addWeighted(this->view.hessians_begin(), view.hessians_cbegin(row), this->view.getNumHessians(),
                                weight);
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::addToSubset(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, const PartialIndexVector& indices,
      StatisticType weight) {
        auto indexIterator = indices.cbegin();
        VectorMath::addToSubsetWeighted(this->view.gradients_begin(), view.gradients_cbegin(row), indexIterator,
                                        this->view.getNumGradients(), weight);
        VectorMath::addToSubsetWeighted(this->view.hessians_begin(), view.hessians_cbegin(row), indexIterator,
                                        this->view.getNumHessians(), weight);
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::difference(
      const DenseDecomposableStatisticVectorView<StatisticType>& first, const CompleteIndexVector& firstIndices,
      const DenseDecomposableStatisticVectorView<StatisticType>& second) {
        VectorMath::difference(this->view.gradients_begin(), first.gradients_cbegin(), second.gradients_cbegin(),
                               this->view.getNumGradients());
        VectorMath::difference(this->view.hessians_begin(), first.hessians_cbegin(), second.hessians_cbegin(),
                               this->view.getNumHessians());
    }

    template<typename StatisticType, typename MemoryAllocator, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, MemoryAllocator, VectorMath>::difference(
      const DenseDecomposableStatisticVectorView<StatisticType>& first, const PartialIndexVector& firstIndices,
      const DenseDecomposableStatisticVectorView<StatisticType>& second) {
        auto indexIterator = firstIndices.cbegin();
        VectorMath::difference(this->view.gradients_begin(), first.gradients_cbegin(), second.gradients_cbegin(),
                               indexIterator, this->view.getNumGradients());
        VectorMath::difference(this->view.hessians_begin(), first.hessians_cbegin(), second.hessians_cbegin(),
                               indexIterator, this->view.getNumHessians());
    }

    template class DenseDecomposableStatisticVector<float32, DefaultMemoryAllocator, SequentialVectorMath>;
    template class DenseDecomposableStatisticVector<float64, DefaultMemoryAllocator, SequentialVectorMath>;

#if SIMD_SUPPORT_ENABLED
    template class DenseDecomposableStatisticVector<float32, SimdMemoryAllocator, SimdVectorMath>;
    template class DenseDecomposableStatisticVector<float64, SimdMemoryAllocator, SimdVectorMath>;
#endif
}
