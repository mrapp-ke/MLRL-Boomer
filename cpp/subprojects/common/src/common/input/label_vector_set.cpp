#include "common/input/label_vector_set.hpp"


template<class T>
typename LabelVectorSet<T>::const_iterator LabelVectorSet<T>::cbegin() const {
    return map_.cbegin();
}

template<class T>
typename LabelVectorSet<T>::const_iterator LabelVectorSet<T>::cend() const {
    return map_.cend();
}

template<class T>
T& LabelVectorSet<T>::addLabelVector(std::unique_ptr<LabelVector> labelVectorPtr) {
    return map_[std::move(labelVectorPtr)];
}

template class LabelVectorSet<uint32>;
