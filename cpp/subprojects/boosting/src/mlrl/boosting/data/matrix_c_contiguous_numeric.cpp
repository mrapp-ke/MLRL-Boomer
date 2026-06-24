#include "mlrl/boosting/data/matrix_c_contiguous_numeric.hpp"

#include "mlrl/common/simd/memory.hpp"

namespace boosting {

    template<typename T, typename MemoryAllocator>
    NumericCContiguousMatrix<T, MemoryAllocator>::NumericCContiguousMatrix(uint32 numRows, uint32 numCols, bool init)
        : DenseMatrixDecorator<CContiguousViewAllocator<CContiguousView<T>, MemoryAllocator>>(
            CContiguousViewAllocator<CContiguousView<T>, MemoryAllocator>(numRows, numCols, init)) {}

    template<typename T, typename MemoryAllocator>
    void NumericCContiguousMatrix<T, MemoryAllocator>::addToRowFromSubset(
      uint32 row, typename View<T>::const_iterator begin, typename View<T>::const_iterator end,
      CompleteIndexVector::const_iterator indicesBegin, CompleteIndexVector::const_iterator indicesEnd) {
        auto iterator = this->values_begin(row);
        uint32 numCols = this->getNumCols();

        for (uint32 i = 0; i < numCols; i++) {
            iterator[i] += begin[i];
        }
    }

    template<typename T, typename MemoryAllocator>
    void NumericCContiguousMatrix<T, MemoryAllocator>::addToRowFromSubset(
      uint32 row, typename View<T>::const_iterator begin, typename View<T>::const_iterator end,
      PartialIndexVector::const_iterator indicesBegin, PartialIndexVector::const_iterator indicesEnd) {
        auto iterator = this->values_begin(row);
        uint32 numCols = indicesEnd - indicesBegin;

        for (uint32 i = 0; i < numCols; i++) {
            uint32 index = indicesBegin[i];
            iterator[index] += begin[i];
        }
    }

    template<typename T, typename MemoryAllocator>
    void NumericCContiguousMatrix<T, MemoryAllocator>::removeFromRowFromSubset(
      uint32 row, typename View<T>::const_iterator begin, typename View<T>::const_iterator end,
      CompleteIndexVector::const_iterator indicesBegin, CompleteIndexVector::const_iterator indicesEnd) {
        auto iterator = this->values_begin(row);
        uint32 numCols = this->getNumCols();

        for (uint32 i = 0; i < numCols; i++) {
            iterator[i] -= begin[i];
        }
    }

    template<typename T, typename MemoryAllocator>
    void NumericCContiguousMatrix<T, MemoryAllocator>::removeFromRowFromSubset(
      uint32 row, typename View<T>::const_iterator begin, typename View<T>::const_iterator end,
      PartialIndexVector::const_iterator indicesBegin, PartialIndexVector::const_iterator indicesEnd) {
        auto iterator = this->values_begin(row);
        uint32 numCols = indicesEnd - indicesBegin;

        for (uint32 i = 0; i < numCols; i++) {
            uint32 index = indicesBegin[i];
            iterator[index] -= begin[i];
        }
    }

    template class NumericCContiguousMatrix<float32, DefaultMemoryAllocator>;
    template class NumericCContiguousMatrix<float64, DefaultMemoryAllocator>;

#if SIMD_SUPPORT_ENABLED
    template class NumericCContiguousMatrix<float32, SimdMemoryAllocator>;
    template class NumericCContiguousMatrix<float64, SimdMemoryAllocator>;
#endif
}
