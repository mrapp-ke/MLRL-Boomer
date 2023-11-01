#include "mlrl/common/data/ring_buffer.hpp"

template<typename T>
RingBuffer<T>::RingBuffer(uint32 capacity)
    : ViewDecorator<AllocatedVector<T>>(AllocatedVector<T>(capacity)), pos_(0), full_(capacity == 0) {}

template<typename T>
typename RingBuffer<T>::const_iterator RingBuffer<T>::cbegin() const {
    return this->view_.array;
}

template<typename T>
typename RingBuffer<T>::const_iterator RingBuffer<T>::cend() const {
    return &this->view_.array[full_ ? this->view_.numElements : pos_];
}

template<typename T>
uint32 RingBuffer<T>::getCapacity() const {
    return this->view_.numElements;
}

template<typename T>
uint32 RingBuffer<T>::getNumElements() const {
    return full_ ? this->view_.numElements : pos_;
}

template<typename T>
bool RingBuffer<T>::isFull() const {
    return full_;
}

template<typename T>
std::pair<bool, T> RingBuffer<T>::push(T value) {
    std::pair<bool, T> result;
    result.first = full_;
    result.second = this->view_.array[pos_];
    this->view_.array[pos_] = value;
    pos_++;

    if (pos_ >= this->view_.numElements) {
        pos_ = 0;
        full_ = true;
    }

    return result;
}

template class RingBuffer<uint8>;
template class RingBuffer<uint32>;
template class RingBuffer<int64>;
template class RingBuffer<float32>;
template class RingBuffer<float64>;
