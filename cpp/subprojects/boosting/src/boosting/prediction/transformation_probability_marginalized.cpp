#include "boosting/prediction/transformation_probability_marginalized.hpp"

#include "common/data/arrays.hpp"
#include "common/math/math.hpp"
#include "probabilities.hpp"

namespace boosting {

    MarginalizedProbabilityTransformation::MarginalizedProbabilityTransformation(
      const LabelVectorSet& labelVectorSet, std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr)
        : labelVectorSet_(labelVectorSet), probabilityFunctionPtr_(std::move(probabilityFunctionPtr)) {}

    void MarginalizedProbabilityTransformation::apply(CContiguousView<float64>::value_iterator scoresBegin,
                                                      CContiguousView<float64>::value_iterator scoresEnd) const {
        uint32 numLabels = scoresEnd - scoresBegin;
        std::pair<std::unique_ptr<DenseVector<float64>>, float64> pair =
          calculateJointProbabilities(scoresBegin, numLabels, labelVectorSet_, *probabilityFunctionPtr_);
        const VectorConstView<float64>& jointProbabilityVector = *pair.first;
        const VectorConstView<float64>::const_iterator jointProbabilityIterator = jointProbabilityVector.cbegin();
        float64 sumOfJointProbabilities = pair.second;
        setArrayToZeros(scoresBegin, numLabels);
        uint32 i = 0;

        for (auto it = labelVectorSet_.cbegin(); it != labelVectorSet_.cend(); it++) {
            const LabelVector& labelVector = *((*it).first);
            uint32 numRelevantLabels = labelVector.getNumElements();
            LabelVector::const_iterator labelIndexIterator = labelVector.cbegin();
            float64 normalizedJointProbability = divideOrZero(jointProbabilityIterator[i], sumOfJointProbabilities);

            for (uint32 j = 0; j < numRelevantLabels; j++) {
                uint32 labelIndex = labelIndexIterator[j];
                scoresBegin[labelIndex] += normalizedJointProbability;
            }

            i++;
        }
    }

}
