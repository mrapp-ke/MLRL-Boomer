#include "boosting/prediction/transformation_probability_marginalized.hpp"

#include "common/data/arrays.hpp"
#include "common/math/math.hpp"
#include "probabilities.hpp"

namespace boosting {

    static inline void calculateMarginalizedProbabilities(CContiguousView<float64>::value_iterator scoreIterator,
                                                          const float64* jointProbabilities,
                                                          float64 sumOfJointProbabilities,
                                                          const LabelVectorSet& labelVectorSet) {
        uint32 i = 0;

        for (auto it = labelVectorSet.cbegin(); it != labelVectorSet.cend(); it++) {
            const auto& entry = *it;
            const std::unique_ptr<LabelVector>& labelVectorPtr = entry.first;
            uint32 numRelevantLabels = labelVectorPtr->getNumElements();
            LabelVector::const_iterator labelIndexIterator = labelVectorPtr->cbegin();
            float64 normalizedJointProbability = divideOrZero(jointProbabilities[i], sumOfJointProbabilities);

            for (uint32 j = 0; j < numRelevantLabels; j++) {
                uint32 labelIndex = labelIndexIterator[j];
                scoreIterator[labelIndex] += normalizedJointProbability;
            }

            i++;
        }
    }

    static inline void predictMarginalizedProbabilities(CContiguousView<float64>::value_iterator scoreIterator,
                                                        uint32 numLabels, const LabelVectorSet& labelVectorSet,
                                                        uint32 numLabelVectors,
                                                        const IProbabilityFunction& probabilityFunction) {
        float64* jointProbabilities = new float64[numLabelVectors];
        float64 sumOfJointProbabilities = calculateJointProbabilities(scoreIterator, numLabels, jointProbabilities,
                                                                      probabilityFunction, labelVectorSet);
        setArrayToZeros(scoreIterator, numLabels);
        calculateMarginalizedProbabilities(scoreIterator, jointProbabilities, sumOfJointProbabilities, labelVectorSet);
        delete[] jointProbabilities;
    }

    MarginalizedProbabilityTransformation::MarginalizedProbabilityTransformation(
      const LabelVectorSet& labelVectorSet, std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr)
        : labelVectorSet_(labelVectorSet), probabilityFunctionPtr_(std::move(probabilityFunctionPtr)) {}

    void MarginalizedProbabilityTransformation::apply(CContiguousView<float64>::value_iterator scoresBegin,
                                                      CContiguousView<float64>::value_iterator scoresEnd) const {
        uint32 numLabels = scoresEnd - scoresBegin;
        uint32 numLabelVectors = labelVectorSet_.getNumLabelVectors();
        predictMarginalizedProbabilities(scoresBegin, numLabels, labelVectorSet_, numLabelVectors,
                                         *probabilityFunctionPtr_);
    }

}
