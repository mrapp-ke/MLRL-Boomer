#include "mlrl/boosting/data/vector_statistic_non_decomposable_dense.hpp"

#include "mlrl/boosting/math/scalar_math.hpp"
#include "mlrl/common/math/vector_math.hpp"
#include "mlrl/common/simd/vector_math.hpp"

namespace boosting {

    template<typename StatisticType, typename VectorMath>
    DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::DenseNonDecomposableStatisticVector(
      uint32 numGradients, bool init)
        : ClearableViewDecorator<
            ViewDecorator<DenseStatisticVectorAllocator<DenseNonDecomposableStatisticVectorView<StatisticType>>>>(
            DenseStatisticVectorAllocator<DenseNonDecomposableStatisticVectorView<StatisticType>>(
              numGradients, math::triangularNumber(numGradients), init)) {}

    template<typename StatisticType, typename VectorMath>
    DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::DenseNonDecomposableStatisticVector(
      const DenseNonDecomposableStatisticVector<StatisticType, VectorMath>& other)
        : DenseNonDecomposableStatisticVector<StatisticType, VectorMath>(other.getNumGradients()) {
        VectorMath::copy(other.view.cbegin(), this->view.begin(), this->view.numElements);
    }

    template<typename StatisticType, typename VectorMath>
    uint32 DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::getNumGradients() const {
        return this->view.getNumGradients();
    }

    template<typename StatisticType, typename VectorMath>
    uint32 DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::getNumHessians() const {
        return this->view.getNumHessians();
    }

    template<typename StatisticType, typename VectorMath>
    void DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::add(
      const DenseNonDecomposableStatisticVectorView<StatisticType>& vector) {
        VectorMath::add(this->view.begin(), vector.cbegin(), this->view.numElements);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::add(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row) {
        VectorMath::add(this->view.begin(), view.values_cbegin(row), this->view.numElements);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::add(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, StatisticType weight) {
        VectorMath::addWeighted(this->view.begin(), view.values_cbegin(row), this->view.numElements, weight);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::remove(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row) {
        VectorMath::subtract(this->view.begin(), view.values_cbegin(row), this->view.numElements);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::remove(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, StatisticType weight) {
        VectorMath::subtractWeighted(this->view.begin(), view.values_cbegin(row), this->view.numElements, weight);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::addToSubset(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, const CompleteIndexVector& indices) {
        VectorMath::add(this->view.begin(), view.values_cbegin(row), this->view.numElements);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::addToSubset(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, const PartialIndexVector& indices) {
        auto indexIterator = indices.cbegin();
        VectorMath::add(this->view.gradients_begin(), view.gradients_cbegin(row), indexIterator,
                        this->getNumGradients());
        auto hessiansBegin = view.hessians_cbegin(row);

        for (uint32 i = 0; i < this->getNumGradients(); i++) {
            uint32 index = indexIterator[i];
            VectorMath::add(&this->view.hessians_begin()[math::triangularNumber(i)],
                            &hessiansBegin[math::triangularNumber(index)], indexIterator, i + 1);
        }
    }

    template<typename StatisticType, typename VectorMath>
    void DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::addToSubset(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, const CompleteIndexVector& indices,
      StatisticType weight) {
        VectorMath::addWeighted(this->view.begin(), view.values_cbegin(row), this->view.numElements, weight);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::addToSubset(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, const PartialIndexVector& indices,
      StatisticType weight) {
        auto indexIterator = indices.cbegin();
        VectorMath::addWeighted(this->view.gradients_begin(), view.gradients_cbegin(row), indexIterator,
                                this->getNumGradients(), weight);
        auto hessiansBegin = view.hessians_cbegin(row);

        for (uint32 i = 0; i < this->getNumGradients(); i++) {
            uint32 index = indexIterator[i];
            VectorMath::addWeighted(&this->view.hessians_begin()[math::triangularNumber(i)],
                                    &hessiansBegin[math::triangularNumber(index)], indexIterator, i + 1, weight);
        }
    }

    template<typename StatisticType, typename VectorMath>
    void DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::difference(
      const DenseNonDecomposableStatisticVectorView<StatisticType>& first, const CompleteIndexVector& firstIndices,
      const DenseNonDecomposableStatisticVectorView<StatisticType>& second) {
        VectorMath::difference(this->view.begin(), first.cbegin(), second.cbegin(), this->view.numElements);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::difference(
      const DenseNonDecomposableStatisticVectorView<StatisticType>& first, const PartialIndexVector& firstIndices,
      const DenseNonDecomposableStatisticVectorView<StatisticType>& second) {
        auto indexIterator = firstIndices.cbegin();
        VectorMath::difference(this->view.gradients_begin(), first.gradients_cbegin(), second.gradients_cbegin(),
                               indexIterator, this->getNumGradients());
        auto firstHessiansBegin = first.hessians_cbegin();
        auto secondHessiansBegin = second.hessians_cbegin();

        for (uint32 i = 0; i < this->getNumGradients(); i++) {
            uint32 offset = math::triangularNumber(i);
            uint32 index = indexIterator[i];
            VectorMath::difference(&this->view.hessians_begin()[offset],
                                   &firstHessiansBegin[math::triangularNumber(index)], &secondHessiansBegin[offset],
                                   indexIterator, i + 1);
        }
    }

    template class DenseNonDecomposableStatisticVector<float32, SequentialVectorMath>;
    template class DenseNonDecomposableStatisticVector<float64, SequentialVectorMath>;

#if SIMD_SUPPORT_ENABLED
    template class DenseNonDecomposableStatisticVector<float32, SimdVectorMath>;
    template class DenseNonDecomposableStatisticVector<float64, SimdVectorMath>;
#endif
}
