#pragma once

#include "common/iterator/binary_forward_iterator.hpp"

namespace boosting {

    static inline float64 calculateJointProbability(LabelVectorSet::const_iterator iterator,
                                                    const float64* scoreIterator, uint32 numLabels,
                                                    const IProbabilityFunction& probabilityFunction) {
        const auto& entry = *iterator;
        const std::unique_ptr<LabelVector>& labelVectorPtr = entry.first;
        auto labelIterator = make_binary_forward_iterator(labelVectorPtr->cbegin(), labelVectorPtr->cend());
        float64 jointProbability = 1;

        for (uint32 i = 0; i < numLabels; i++) {
            float64 score = scoreIterator[i];
            float64 probability = probabilityFunction.transform(score);
            bool trueLabel = *labelIterator;

            if (!trueLabel) {
                probability = 1 - probability;
            }

            jointProbability *= probability;
            labelIterator++;
        }

        return jointProbability;
    }

    static inline float64 calculateJointProbabilities(const float64* scoreIterator, uint32 numLabels,
                                                      float64* jointProbabilities,
                                                      const IProbabilityFunction& probabilityFunction,
                                                      const LabelVectorSet& labelVectorSet) {
        LabelVectorSet::const_iterator it = labelVectorSet.cbegin();
        float64 sumOfJointProbabilities = calculateJointProbability(it, scoreIterator, numLabels, probabilityFunction);
        jointProbabilities[0] = sumOfJointProbabilities;
        it++;
        uint32 i = 1;

        for (; it != labelVectorSet.cend(); it++) {
            float64 jointProbability = calculateJointProbability(it, scoreIterator, numLabels, probabilityFunction);
            sumOfJointProbabilities += jointProbability;
            jointProbabilities[i] = jointProbability;
            i++;
        }

        return sumOfJointProbabilities;
    }

}
