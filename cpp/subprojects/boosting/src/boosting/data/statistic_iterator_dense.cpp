#include "boosting/data/statistic_iterator_dense.hpp"


namespace boosting {

    DenseGradientConstIterator::DenseGradientConstIterator(const Tuple<float64>* ptr)
        : ptr_(ptr) {

    }

    DenseGradientConstIterator::reference DenseGradientConstIterator::operator[](uint32 index) const {
        return ptr_[index].first;
    }

    DenseGradientConstIterator::reference DenseGradientConstIterator::operator*() const {
        return (*ptr_).first;
    }

    DenseGradientConstIterator& DenseGradientConstIterator::operator++() {
        ++ptr_;
        return *this;
    }

    DenseGradientConstIterator& DenseGradientConstIterator::operator++(int n) {
        ptr_++;
        return *this;
    }

    DenseGradientConstIterator& DenseGradientConstIterator::operator--() {
        --ptr_;
        return *this;
    }

    DenseGradientConstIterator& DenseGradientConstIterator::operator--(int n) {
        ptr_--;
        return *this;
    }

    bool DenseGradientConstIterator::operator!=(const DenseGradientConstIterator& rhs) const {
        return ptr_ != rhs.ptr_;
    }

    bool DenseGradientConstIterator::operator==(const DenseGradientConstIterator& rhs) const {
        return ptr_ == rhs.ptr_;
    }

    DenseGradientConstIterator::difference_type DenseGradientConstIterator::operator-(
            const DenseGradientConstIterator& rhs) const {
        return (difference_type) (ptr_ - rhs.ptr_);
    }

    DenseHessianConstIterator::DenseHessianConstIterator(const Tuple<float64>* ptr)
        : ptr_(ptr) {

    }

    DenseHessianConstIterator::reference DenseHessianConstIterator::operator[](uint32 index) const {
        return ptr_[index].second;
    }

    DenseHessianConstIterator::reference DenseHessianConstIterator::operator*() const {
        return (*ptr_).second;
    }

    DenseHessianConstIterator& DenseHessianConstIterator::operator++() {
        ++ptr_;
        return *this;
    }

    DenseHessianConstIterator& DenseHessianConstIterator::operator++(int n) {
        ptr_++;
        return *this;
    }

    DenseHessianConstIterator& DenseHessianConstIterator::operator--() {
        --ptr_;
        return *this;
    }

    DenseHessianConstIterator& DenseHessianConstIterator::operator--(int n) {
        ptr_--;
        return *this;
    }

    bool DenseHessianConstIterator::operator!=(const DenseHessianConstIterator& rhs) const {
        return ptr_ != rhs.ptr_;
    }

    bool DenseHessianConstIterator::operator==(const DenseHessianConstIterator& rhs) const {
        return ptr_ == rhs.ptr_;
    }

    DenseHessianConstIterator::difference_type DenseHessianConstIterator::operator-(const DenseHessianConstIterator& rhs) const {
        return (difference_type) (ptr_ - rhs.ptr_);
    }

}
