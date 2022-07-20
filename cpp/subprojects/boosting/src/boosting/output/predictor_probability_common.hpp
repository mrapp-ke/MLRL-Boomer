/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/iterator/binary_forward_iterator.hpp"


namespace boosting {

    static inline float64 calculateJointProbability(LabelVectorSet::const_iterator iterator, const float64* scoresBegin,
                                                    const float64* scoresEnd,
                                                    const IProbabilityFunction& probabilityFunction) {
        const auto& entry = *iterator;
        const std::unique_ptr<LabelVector>& labelVectorPtr = entry.first;
        auto labelIterator = make_binary_forward_iterator(labelVectorPtr->cbegin(), labelVectorPtr->cend());
        uint32 numElements = scoresEnd - scoresBegin;
        float64 jointProbability = 1;

        for (uint32 i = 0; i < numElements; i++) {
            float64 score = scoresBegin[i];
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

    static inline float64 calculateJointProbabilities(const float64* scoresBegin, const float64* scoresEnd,
                                                      float64* jointProbabilities,
                                                      const IProbabilityFunction& probabilityFunction,
                                                      const LabelVectorSet& labelVectorSet) {
        LabelVectorSet::const_iterator it = labelVectorSet.cbegin();
        float64 sumOfJointProbabilities = calculateJointProbability(it, scoresBegin, scoresEnd, probabilityFunction);
        jointProbabilities[0] = sumOfJointProbabilities;
        it++;
        uint32 i = 1;

        for (; it != labelVectorSet.cend(); it++) {
            float64 jointProbability = calculateJointProbability(it, scoresBegin, scoresEnd, probabilityFunction);
            sumOfJointProbabilities += jointProbability;
            jointProbabilities[i] = jointProbability;
            i++;
        }

        return sumOfJointProbabilities;
    }

}
