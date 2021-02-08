#include "common/sampling/instance_sampling_no.hpp"
#include "common/sampling/weight_vector_equal.hpp"
#include "common/sampling/weight_vector_dense.hpp"


std::unique_ptr<IWeightVector> NoInstanceSubSampling::subSample(uint32 numExamples, RNG& rng) const {
    return std::make_unique<EqualWeightVector>(numExamples);
}

std::unique_ptr<IWeightVector> NoInstanceSubSampling::subSample(std::unique_ptr<SinglePartition> partitionPtr,
                                                                RNG& rng) const {
    return std::make_unique<EqualWeightVector>(partitionPtr->getNumElements());
}

std::unique_ptr<IWeightVector> NoInstanceSubSampling::subSample(std::unique_ptr<BiPartition> partitionPtr,
                                                                RNG& rng) const {
    uint32 numExamples = partitionPtr->getNumElements();
    uint32 numTrainingExamples = partitionPtr->getNumFirst();
    BiPartition::const_iterator indexIterator = partitionPtr->first_cbegin();
    std::unique_ptr<DenseWeightVector> weightVectorPtr = std::make_unique<DenseWeightVector>(numExamples,
                                                                                             numTrainingExamples);
    DenseWeightVector::iterator weightIterator = weightVectorPtr->begin();

    for (uint32 i = 0; i < numTrainingExamples; i++) {
        uint32 index = indexIterator[i];
        weightIterator[index] = 1;
    }

    return std::move(weightVectorPtr);
}
