#include "common/data/vector_dok.hpp"

template<class T>
DokVector<T>::DokVector(T sparseValue)
    : sparseValue_(sparseValue) {

}

template<class T>
T DokVector<T>::getValue(uint32 pos) const {
    auto it = data_.find(pos);
    return it != data_.cend() ? it->second : sparseValue_;
}

template<class T>
void DokVector<T>::setValue(uint32 pos, T value) {
    auto result = data_.emplace(pos, value);

    if (!result.second) {
        result.first->second = value;
    }
}

template<class T>
void DokVector<T>::setAllToZero() {
    data_.clear();
}

template class DokVector<uint32>;
