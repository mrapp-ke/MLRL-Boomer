#include "mlrl/seco/data/vector_statistic_decomposable_dense.hpp"

#include "mlrl/common/iterator/iterator_forward_sparse_binary.hpp"
#include "mlrl/common/math/vector_math.hpp"
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

    template<typename StatisticType>
    DenseDecomposableStatisticVectorView<StatisticType>::DenseDecomposableStatisticVectorView(uint32 numElements,
                                                                                              bool init)
        : CompositeVector<AllocatedVector<StatisticType>, AllocatedVector<StatisticType>>(
            AllocatedVector<StatisticType>(numElements, init), AllocatedVector<StatisticType>(numElements, init)) {}

    template<typename StatisticType>
    typename View<StatisticType>::const_iterator
      DenseDecomposableStatisticVectorView<StatisticType>::correct_indices_cbegin() const {
        return this->firstView.cbegin();
    }

    template<typename StatisticType>
    typename View<StatisticType>::const_iterator
      DenseDecomposableStatisticVectorView<StatisticType>::correct_indices_cend() const {
        return this->firstView.cend();
    }

    template<typename StatisticType>
    typename View<StatisticType>::iterator
      DenseDecomposableStatisticVectorView<StatisticType>::correct_indices_begin() {
        return this->firstView.begin();
    }

    template<typename StatisticType>
    typename View<StatisticType>::iterator DenseDecomposableStatisticVectorView<StatisticType>::correct_indices_end() {
        return this->firstView.end();
    }

    template<typename StatisticType>
    typename View<StatisticType>::const_iterator
      DenseDecomposableStatisticVectorView<StatisticType>::incorrect_indices_cbegin() const {
        return this->secondView.cbegin();
    }

    template<typename StatisticType>
    typename View<StatisticType>::const_iterator
      DenseDecomposableStatisticVectorView<StatisticType>::incorrect_indices_cend() const {
        return this->secondView.cend();
    }

    template<typename StatisticType>
    typename View<StatisticType>::iterator
      DenseDecomposableStatisticVectorView<StatisticType>::incorrect_indices_begin() {
        return this->secondView.begin();
    }

    template<typename StatisticType>
    typename View<StatisticType>::iterator
      DenseDecomposableStatisticVectorView<StatisticType>::incorrect_indices_end() {
        return this->secondView.end();
    }

    template<typename StatisticType>
    uint32 DenseDecomposableStatisticVectorView<StatisticType>::getNumElements() const {
        return this->firstView.numElements;
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVectorView<StatisticType>::clear() {
        this->firstView.clear();
        this->secondView.clear();
    }

    template class DenseDecomposableStatisticVectorView<uint32>;
    template class DenseDecomposableStatisticVectorView<float32>;

    template<typename StatisticType, typename VectorMath>
    DenseDecomposableStatisticVector<StatisticType, VectorMath>::DenseDecomposableStatisticVector(uint32 numElements,
                                                                                                  bool init)
        : ClearableViewDecorator<ViewDecorator<DenseDecomposableStatisticVectorView<StatisticType>>>(
            DenseDecomposableStatisticVectorView<StatisticType>(numElements, init)) {}

    template<typename StatisticType, typename VectorMath>
    DenseDecomposableStatisticVector<StatisticType, VectorMath>::DenseDecomposableStatisticVector(
      const DenseDecomposableStatisticVector<StatisticType, VectorMath>& other)
        : DenseDecomposableStatisticVector(other.getNumElements()) {
        uint32 numElements = this->getNumElements();
        VectorMath::copy(other.correct_indices_cbegin(), this->correct_indices_begin(), numElements);
        VectorMath::copy(other.incorrect_indices_cbegin(), this->incorrect_indices_begin(), numElements);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, VectorMath>::add(
      const DenseDecomposableStatisticVectorView<StatisticType>& other) {
        uint32 numElements = this->getNumElements();
        VectorMath::add(this->correct_indices_begin(), other.correct_indices_cbegin(), numElements);
        VectorMath::add(this->incorrect_indices_begin(), other.incorrect_indices_cbegin(), numElements);
    }

    template<typename StatisticType, typename VectorMath>
    typename View<StatisticType>::const_iterator
      DenseDecomposableStatisticVector<StatisticType, VectorMath>::correct_indices_cbegin() const {
        return this->view.correct_indices_cbegin();
    }

    template<typename StatisticType, typename VectorMath>
    typename View<StatisticType>::const_iterator
      DenseDecomposableStatisticVector<StatisticType, VectorMath>::correct_indices_cend() const {
        return this->view.correct_indices_cend();
    }

    template<typename StatisticType, typename VectorMath>
    typename View<StatisticType>::iterator
      DenseDecomposableStatisticVector<StatisticType, VectorMath>::correct_indices_begin() {
        return this->view.correct_indices_begin();
    }

    template<typename StatisticType, typename VectorMath>
    typename View<StatisticType>::iterator
      DenseDecomposableStatisticVector<StatisticType, VectorMath>::correct_indices_end() {
        return this->view.correct_indices_end();
    }

    template<typename StatisticType, typename VectorMath>
    typename View<StatisticType>::const_iterator
      DenseDecomposableStatisticVector<StatisticType, VectorMath>::incorrect_indices_cbegin() const {
        return this->view.incorrect_indices_cbegin();
    }

    template<typename StatisticType, typename VectorMath>
    typename View<StatisticType>::const_iterator
      DenseDecomposableStatisticVector<StatisticType, VectorMath>::incorrect_indices_cend() const {
        return this->view.incorrect_indices_cend();
    }

    template<typename StatisticType, typename VectorMath>
    typename View<StatisticType>::iterator
      DenseDecomposableStatisticVector<StatisticType, VectorMath>::incorrect_indices_begin() {
        return this->view.incorrect_indices_begin();
    }

    template<typename StatisticType, typename VectorMath>
    typename View<StatisticType>::iterator
      DenseDecomposableStatisticVector<StatisticType, VectorMath>::incorrect_indices_end() {
        return this->view.incorrect_indices_end();
    }

    template<typename StatisticType, typename VectorMath>
    uint32 DenseDecomposableStatisticVector<StatisticType, VectorMath>::getNumElements() const {
        return this->view.getNumElements();
    }

    template<typename StatisticType, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, VectorMath>::add(const SparseDecomposableStatisticView& view,
                                                                          uint32 row, StatisticType weight) {
        VectorMath::addConstantToSubset(this->correct_indices_begin(), weight, view.correct_indices_cbegin(row),
                                        view.correct_indices_cend(row) - view.correct_indices_cbegin(row));
        VectorMath::addConstantToSubset(this->incorrect_indices_begin(), weight, view.incorrect_indices_cbegin(row),
                                        view.incorrect_indices_cend(row) - view.incorrect_indices_cbegin(row));
    }

    template<typename StatisticType, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, VectorMath>::remove(
      const SparseDecomposableStatisticView& view, uint32 row, StatisticType weight) {
        VectorMath::subtractConstantFromSubset(this->correct_indices_begin(), weight, view.correct_indices_cbegin(row),
                                               view.correct_indices_cend(row) - view.correct_indices_cbegin(row));
        VectorMath::subtractConstantFromSubset(this->incorrect_indices_begin(), weight,
                                               view.incorrect_indices_cbegin(row),
                                               view.incorrect_indices_cend(row) - view.incorrect_indices_cbegin(row));
    }

    template<typename StatisticType, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, VectorMath>::addToSubset(
      const SparseDecomposableStatisticView& view, uint32 row, const CompleteIndexVector& indices,
      StatisticType weight) {
        VectorMath::addConstantToSubset(this->correct_indices_begin(), weight, view.correct_indices_cbegin(row),
                                        view.correct_indices_cend(row) - view.correct_indices_cbegin(row));
        VectorMath::addConstantToSubset(this->incorrect_indices_begin(), weight, view.incorrect_indices_cbegin(row),
                                        view.incorrect_indices_cend(row) - view.incorrect_indices_cbegin(row));
    }

    template<typename StatisticType, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, VectorMath>::addToSubset(
      const SparseDecomposableStatisticView& view, uint32 row, const PartialIndexVector& indices,
      StatisticType weight) {
        addConstantToSubset(this->correct_indices_begin(), weight, view.correct_indices_cbegin(row),
                            view.correct_indices_cend(row) - view.correct_indices_cbegin(row), indices);
        addConstantToSubset(this->incorrect_indices_begin(), weight, view.incorrect_indices_cbegin(row),
                            view.incorrect_indices_cend(row) - view.incorrect_indices_cbegin(row), indices);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, VectorMath>::difference(
      const DenseDecomposableStatisticVectorView<StatisticType>& first, const CompleteIndexVector& firstIndices,
      const DenseDecomposableStatisticVectorView<StatisticType>& second) {
        uint32 numElements = this->getNumElements();
        VectorMath::difference(this->correct_indices_begin(), first.correct_indices_cbegin(),
                               second.correct_indices_cbegin(), numElements);
        VectorMath::difference(this->incorrect_indices_begin(), first.incorrect_indices_cbegin(),
                               second.incorrect_indices_cbegin(), numElements);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, VectorMath>::difference(
      const DenseDecomposableStatisticVectorView<StatisticType>& first, const PartialIndexVector& firstIndices,
      const DenseDecomposableStatisticVectorView<StatisticType>& second) {
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        uint32 numElements = this->getNumElements();
        VectorMath::difference(this->correct_indices_begin(), first.correct_indices_cbegin(),
                               second.correct_indices_cbegin(), indexIterator, numElements);
        VectorMath::difference(this->incorrect_indices_begin(), first.incorrect_indices_cbegin(),
                               second.incorrect_indices_cbegin(), indexIterator, numElements);
    }

    template class DenseDecomposableStatisticVector<uint32, SequentialVectorMath>;
    template class DenseDecomposableStatisticVector<float32, SequentialVectorMath>;

#if SIMD_SUPPORT_ENABLED
    template class DenseDecomposableStatisticVector<uint32, SimdVectorMath>;
    template class DenseDecomposableStatisticVector<float32, SimdVectorMath>;
#endif
}
