#include "mlrl/boosting/data/vector_statistic_decomposable_bit.hpp"

namespace boosting {

    static inline void copyInternally(const BitVector<uint32>& firstView, BitVector<uint32>& secondView) {
        typename BitVector<uint32>::const_iterator begin = firstView.cbegin();
        uint32 arraySize = firstView.cend() - begin;
        util::copyView(begin, secondView.begin(), arraySize);
    }

    static inline void addInternally(BitVector<uint32>& firstView, const BitVector<uint32>& secondView) {
        typename BitVector<uint32>::iterator begin = firstView.begin();
        uint32 arraySize = firstView.end() - begin;
        util::addToView(begin, secondView.cbegin(), arraySize);
    }

    static inline void addInternally(BitVector<uint32>& firstView, const BitVector<uint32>& secondView,
                                     const PartialIndexVector& indices) {
        uint32 numIndices = indices.getNumElements();

        for (uint32 i = 0; i < numIndices; i++) {
            uint32 index = indices[i];
            uint32 value = secondView[index];
            firstView.set(i, value);
        }
    }

    static inline void removeInternally(BitVector<uint32>& firstView, const BitVector<uint32>& secondView) {
        typename BitVector<uint32>::iterator begin = firstView.begin();
        uint32 arraySize = firstView.end() - begin;
        util::removeFromView(begin, secondView.cbegin(), arraySize);
    }

    static inline void differenceInternally(BitVector<uint32>& firstView, const BitVector<uint32>& secondView,
                                            const BitVector<uint32>& thirdView) {
        typename BitVector<uint32>::iterator begin = firstView.begin();
        uint32 arraySize = firstView.end() - begin;
        util::setViewToDifference(begin, secondView.cbegin(), thirdView.cbegin(), arraySize);
    }

    static inline void differenceInternally(BitVector<uint32>& firstView, const BitVector<uint32>& secondView,
                                            const PartialIndexVector& indices, const BitVector<uint32>& thirdView) {
        uint32 numIndices = indices.getNumElements();

        for (uint32 i = 0; i < numIndices; i++) {
            uint32 index = indices[i];
            uint32 firstValue = secondView[index];
            uint32 secondValue = thirdView[i];
            firstView.set(i, firstValue - secondValue);
        }
    }

    BitDecomposableStatisticVector::BitDecomposableStatisticVector(const BitDecomposableStatisticView& view,
                                                                   uint32 numElements, bool init)
        : CompositeView<AllocatedBitVector<uint32>, AllocatedBitVector<uint32>>(
            AllocatedBitVector<uint32>(numElements, view.firstView.numBitsPerElement, init),
            AllocatedBitVector<uint32>(numElements, view.secondView.numBitsPerElement, init)) {}

    BitDecomposableStatisticVector::BitDecomposableStatisticVector(const BitDecomposableStatisticVector& other)
        : CompositeView<AllocatedBitVector<uint32>, AllocatedBitVector<uint32>>(
            AllocatedBitVector<uint32>(other.firstView.numElements, other.firstView.numBitsPerElement),
            AllocatedBitVector<uint32>(other.secondView.numElements, other.secondView.numBitsPerElement)) {
        copyInternally(other.firstView, this->firstView);
        copyInternally(other.secondView, this->secondView);
    }

    uint32 BitDecomposableStatisticVector::getNumElements() const {
        return this->firstView.numElements;
    }

    uint32 BitDecomposableStatisticVector::getNumBitsPerElement() const {
        return this->firstView.numBitsPerElement;
    }

    void BitDecomposableStatisticVector::add(const BitDecomposableStatisticVector& vector) {
        addInternally(this->firstView, vector.firstView);
        addInternally(this->secondView, vector.secondView);
    }

    void BitDecomposableStatisticVector::add(const BitDecomposableStatisticView& view, uint32 row) {
        addInternally(this->firstView, view.firstView[row]);
        addInternally(this->secondView, view.secondView[row]);
    }

    void BitDecomposableStatisticVector::add(const BitDecomposableStatisticView& view, uint32 row, float64 weight) {
        // TODO Implement
        throw std::runtime_error("not implemented");
    }

    void BitDecomposableStatisticVector::remove(const BitDecomposableStatisticView& view, uint32 row) {
        removeInternally(this->firstView, view.firstView[row]);
        removeInternally(this->secondView, view.secondView[row]);
    }

    void BitDecomposableStatisticVector::remove(const BitDecomposableStatisticView& view, uint32 row, float64 weight) {
        // TODO Implement
        throw std::runtime_error("not implemented");
    }

    void BitDecomposableStatisticVector::addToSubset(const BitDecomposableStatisticView& view, uint32 row,
                                                     const CompleteIndexVector& indices) {
        addInternally(this->firstView, view.firstView[row]);
        addInternally(this->secondView, view.secondView[row]);
    }

    void BitDecomposableStatisticVector::addToSubset(const BitDecomposableStatisticView& view, uint32 row,
                                                     const PartialIndexVector& indices) {
        addInternally(this->firstView, view.firstView[row], indices);
        addInternally(this->secondView, view.secondView[row], indices);
    }

    void BitDecomposableStatisticVector::addToSubset(const BitDecomposableStatisticView& view, uint32 row,
                                                     const CompleteIndexVector& indices, float64 weight) {
        // TODO Implement
        throw std::runtime_error("not implemented");
    }

    void BitDecomposableStatisticVector::addToSubset(const BitDecomposableStatisticView& view, uint32 row,
                                                     const PartialIndexVector& indices, float64 weight) {
        // TODO Implement
        throw std::runtime_error("not implemented");
    }

    void BitDecomposableStatisticVector::difference(const BitDecomposableStatisticVector& first,
                                                    const CompleteIndexVector& firstIndices,
                                                    const BitDecomposableStatisticVector& second) {
        differenceInternally(this->firstView, first.firstView, second.firstView);
        differenceInternally(this->secondView, first.secondView, second.secondView);
    }

    void BitDecomposableStatisticVector::difference(const BitDecomposableStatisticVector& first,
                                                    const PartialIndexVector& firstIndices,
                                                    const BitDecomposableStatisticVector& second) {
        differenceInternally(this->firstView, first.firstView, firstIndices, second.firstView);
        differenceInternally(this->secondView, first.secondView, firstIndices, second.secondView);
    }

    void BitDecomposableStatisticVector::clear() {
        this->firstView.clear();
        this->secondView.clear();
    }

}
