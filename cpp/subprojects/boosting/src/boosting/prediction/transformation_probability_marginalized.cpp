#include "boosting/prediction/transformation_probability_marginalized.hpp"

#include "common/data/arrays.hpp"
#include "common/data/vector_dense.hpp"
#include "common/math/math.hpp"
#include "probabilities.hpp"

namespace boosting {

    MarginalizedProbabilityTransformation::MarginalizedProbabilityTransformation(
      const LabelVectorSet& labelVectorSet, std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr)
        : labelVectorSet_(labelVectorSet), probabilityFunctionPtr_(std::move(probabilityFunctionPtr)) {}

    void MarginalizedProbabilityTransformation::apply(CContiguousView<float64>::value_iterator scoresBegin,
                                                      CContiguousView<float64>::value_iterator scoresEnd) const {
        uint32 numLabels = scoresEnd - scoresBegin;
        uint32 numLabelVectors = labelVectorSet_.getNumLabelVectors();
        DenseVector<float64> jointProbabilityVector(numLabelVectors);
        DenseVector<float64>::iterator jointProbabilityIterator = jointProbabilityVector.begin();
        float64 sumOfJointProbabilities = calculateJointProbabilities(scoresBegin, numLabels, jointProbabilityIterator,
                                                                      *probabilityFunctionPtr_, labelVectorSet_);
        setArrayToZeros(scoresBegin, numLabels);
        uint32 i = 0;

        for (auto it = labelVectorSet_.cbegin(); it != labelVectorSet_.cend(); it++) {
            const auto& entry = *it;
            const std::unique_ptr<LabelVector>& labelVectorPtr = entry.first;
            uint32 numRelevantLabels = labelVectorPtr->getNumElements();
            LabelVector::const_iterator labelIndexIterator = labelVectorPtr->cbegin();
            float64 normalizedJointProbability = divideOrZero(jointProbabilityIterator[i], sumOfJointProbabilities);

            for (uint32 j = 0; j < numRelevantLabels; j++) {
                uint32 labelIndex = labelIndexIterator[j];
                scoresBegin[labelIndex] += normalizedJointProbability;
            }

            i++;
        }
    }

}
