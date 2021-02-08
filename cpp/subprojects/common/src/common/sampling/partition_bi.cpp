#include "common/sampling/partition_bi.hpp"


BiPartition::BiPartition(uint32 numFirst, uint32 numSecond)
    : vector_(DenseVector<uint32>(numFirst + numSecond)), numFirst_(numFirst) {

}

BiPartition::iterator BiPartition::first_begin() {
    return vector_.begin();
}

BiPartition::iterator BiPartition::first_end() {
    return &vector_.begin()[numFirst_];
}

BiPartition::const_iterator BiPartition::first_cbegin() const {
    return vector_.cbegin();
}

BiPartition::const_iterator BiPartition::first_cend() const {
    return &vector_.cbegin()[numFirst_];
}

BiPartition::iterator BiPartition::second_begin() {
    return &vector_.begin()[numFirst_];
}

BiPartition::iterator BiPartition::second_end() {
    return vector_.end();
}

BiPartition::const_iterator BiPartition::second_cbegin() const {
    return &vector_.cbegin()[numFirst_];
}

BiPartition::const_iterator BiPartition::second_cend() const {
    return vector_.cend();
}

uint32 BiPartition::getNumFirst() const {
    return numFirst_;
}

uint32 BiPartition::getNumSecond() const {
    return vector_.getNumElements() - numFirst_;
}
