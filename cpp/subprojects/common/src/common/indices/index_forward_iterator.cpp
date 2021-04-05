#include "common/indices/index_forward_iterator.hpp"


template<class T>
IndexForwardIterator<T>::IndexForwardIterator(T begin, T end)
    : iterator_(begin), end_(end), index_(0) {

}

template<class T>
typename IndexForwardIterator<T>::reference IndexForwardIterator<T>::operator*() const {
    return iterator_ != end_ && *iterator_ == index_;
}

template<class T>
IndexForwardIterator<T>& IndexForwardIterator<T>::operator++() {
    ++index_;

    if (iterator_ != end_) {
        iterator_++;
    }

    return *this;
}

template<class T>
IndexForwardIterator<T>& IndexForwardIterator<T>::operator++(int n) {
    index_++;

    if (iterator_ != end_) {
        iterator_++;
    }

    return *this;
}

template<class T>
bool IndexForwardIterator<T>::operator!=(const IndexForwardIterator<T>& rhs) const {
    return index_ != rhs.index_;
}
