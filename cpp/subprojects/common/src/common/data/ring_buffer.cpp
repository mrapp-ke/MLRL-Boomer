#include "common/data/ring_buffer.hpp"


template<class T>
RingBuffer<T>::RingBuffer(uint32 capacity)
    : array_(new T[capacity]), capacity_(capacity), pos_(0), full_(false) {

}

template<class T>
RingBuffer<T>::~RingBuffer() {
    delete[] array_;
}

template<class T>
typename RingBuffer<T>::const_iterator RingBuffer<T>::cbegin() const {
    return array_;
}

template<class T>
typename RingBuffer<T>::const_iterator RingBuffer<T>::cend() const {
    return &array_[full_ ? capacity_ : pos_];
}

template<class T>
uint32 RingBuffer<T>::getCapacity() const {
    return capacity_;
}

template<class T>
uint32 RingBuffer<T>::getNumElements() const {
    return full_ ? capacity_ : pos_;
}

template<class T>
void RingBuffer<T>::push(T value) {
    array_[pos_] = value;
    pos_++;

    if (pos_ >= capacity_) {
        pos_ = 0;
        full_ = true;
    }
}

template class RingBuffer<float64>;
