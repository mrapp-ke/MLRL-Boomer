#include "common/input/label_vector.hpp"
#include <cstdlib>


LabelVector::LabelVector(uint32 numElements)
    : LabelVector(numElements, false) {

}

LabelVector::LabelVector(uint32 numElements, bool init)
    : numElements_(numElements), maxCapacity_(numElements),
      array_((uint32*) (init ? calloc(numElements, sizeof(uint32)) : malloc(numElements * sizeof(uint32)))) {

}

LabelVector::index_iterator LabelVector::indices_begin() {
    return array_;
}

LabelVector::index_iterator LabelVector::indices_end() {
    return &array_[numElements_];
}

LabelVector::index_const_iterator LabelVector::indices_cbegin() const {
    return array_;
}

LabelVector::index_const_iterator LabelVector::indices_cend() const {
    return &array_[numElements_];
}

uint32 LabelVector::getNumElements() const {
    return numElements_;
}

void LabelVector::setNumElements(uint32 numElements, bool freeMemory) {
    if (numElements < maxCapacity_) {
        if (freeMemory) {
            array_ = (uint32*) realloc(array_, numElements * sizeof(uint32));
            maxCapacity_ = numElements;
        }
    } else if (numElements > maxCapacity_) {
        array_ = (uint32*) realloc(array_, numElements * sizeof(uint32));
        maxCapacity_ = numElements;
    }

    numElements_ = numElements;
}
