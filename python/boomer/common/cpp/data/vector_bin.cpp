#include "vector_bin.h"


BinVector::BinVector(uint32 numElements)
    : DenseVector<Bin>(numElements, true) {

}

BinVector::example_const_iterator BinVector::examples_cbegin(uint32 binIndex) {
    std::forward_list<Example>& examples = examplesPerBin_[binIndex];
    return examples.cbegin();
}

BinVector::example_const_iterator BinVector::examples_cend(uint32 binIndex) {
    std::forward_list<Example>& examples = examplesPerBin_[binIndex];
    return examples.cend();
}

void BinVector::addExample(uint32 binIndex, Example example) {
    std::forward_list<Example>& examples = examplesPerBin_[binIndex];
    examples.push_front(example);
}

void BinVector::clearAllExamples() {
    examplesPerBin_.clear();
}
