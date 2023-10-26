#include "mlrl/common/data/vector_bit.hpp"

#include "mlrl/common/util/view_functions.hpp"

#include <climits>

constexpr std::size_t UINT32_SIZE = CHAR_BIT * sizeof(uint32);

static inline constexpr std::size_t size(uint32 numElements) {
    return (numElements + UINT32_SIZE - 1) / UINT32_SIZE;
}

static inline constexpr uint32 index(uint32 pos) {
    return pos / UINT32_SIZE;
}

static inline constexpr uint32 mask(uint32 pos) {
    return 1U << (pos % UINT32_SIZE);
}

BitVector::BitVector(uint32 numElements) : BitVector(numElements, false) {}

BitVector::BitVector(uint32 numElements, bool init)
    : VectorDecorator<AllocatedView<View<uint32>>>(AllocatedView<View<uint32>>(size(numElements), init)),
      numElements_(numElements) {}

bool BitVector::operator[](uint32 pos) const {
    return this->view_.array[index(pos)] & mask(pos);
}

void BitVector::set(uint32 pos, bool value) {
    if (value) {
        this->view_.array[index(pos)] |= mask(pos);
    } else {
        this->view_.array[index(pos)] &= ~mask(pos);
    }
}

uint32 BitVector::getNumElements() const {
    return numElements_;
}

void BitVector::clear() {
    setViewToZeros(this->view_.array, size(numElements_));
}
