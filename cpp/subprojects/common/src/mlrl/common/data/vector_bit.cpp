#include "mlrl/common/data/vector_bit.hpp"

#include <climits>

static inline constexpr uint32 UINT32_SIZE = static_cast<uint32>(CHAR_BIT * sizeof(uint32));

static inline constexpr uint32 size(uint32 numElements) {
    return numElements / UINT32_SIZE + (numElements % UINT32_SIZE != 0);
}

static inline constexpr uint32 index(uint32 pos) {
    return pos / UINT32_SIZE;
}

static inline constexpr uint32 mask(uint32 pos) {
    return 1U << (pos % UINT32_SIZE);
}

BitVector::BitVector(uint32 numElements, bool init)
    : ClearableViewDecorator<VectorDecorator<AllocatedVector<uint32>>>(
        AllocatedVector<uint32>(size(numElements), init)),
      numElements_(numElements) {}

bool BitVector::operator[](uint32 pos) const {
    return this->view.array[index(pos)] & mask(pos);
}

void BitVector::set(uint32 pos, bool value) {
    if (value) {
        this->view.array[index(pos)] |= mask(pos);
    } else {
        this->view.array[index(pos)] &= ~mask(pos);
    }
}

uint32 BitVector::getNumElements() const {
    return numElements_;
}
