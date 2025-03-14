#include "mlrl/common/iterator/iterator_index.hpp"

IndexIterator::IndexIterator() : IndexIterator(0) {}

IndexIterator::IndexIterator(uint32 index) : index_(index) {}

IndexIterator::value_type IndexIterator::operator[](uint32 index) const {
    return index;
}

IndexIterator::value_type IndexIterator::operator*() const {
    return index_;
}

IndexIterator& IndexIterator::operator++() {
    ++index_;
    return *this;
}

IndexIterator& IndexIterator::operator++(int n) {
    index_++;
    return *this;
}

IndexIterator& IndexIterator::operator--() {
    --index_;
    return *this;
}

IndexIterator& IndexIterator::operator--(int n) {
    index_--;
    return *this;
}

bool IndexIterator::operator!=(const IndexIterator& rhs) const {
    return index_ != rhs.index_;
}

bool IndexIterator::operator==(const IndexIterator& rhs) const {
    return index_ == rhs.index_;
}

IndexIterator::difference_type IndexIterator::operator-(const IndexIterator& rhs) const {
    return (difference_type) index_ - (difference_type) rhs.index_;
}
