#include "boosting/prediction/transformation_probability_marginalized.hpp"

#include "common/data/arrays.hpp"
#include "joint_probabilities.hpp"

namespace boosting {

    MarginalizedProbabilityTransformation::MarginalizedProbabilityTransformation(
      const LabelVectorSet& labelVectorSet,
      std::unique_ptr<IMarginalProbabilityFunction> marginalProbabilityFunctionPtr)
        : labelVectorSet_(labelVectorSet), marginalProbabilityFunctionPtr_(std::move(marginalProbabilityFunctionPtr)) {}

    void MarginalizedProbabilityTransformation::apply(VectorConstView<float64>::const_iterator scoresBegin,
                                                      VectorConstView<float64>::const_iterator scoresEnd,
                                                      VectorView<float64>::iterator probabilitiesBegin,
                                                      VectorView<float64>::iterator probabilitiesEnd) const {
        uint32 numLabels = scoresEnd - scoresBegin;
        std::pair<std::unique_ptr<DenseVector<float64>>, float64> pair =
          calculateJointProbabilities(scoresBegin, numLabels, labelVectorSet_, *marginalProbabilityFunctionPtr_);
        const VectorConstView<float64>& jointProbabilityVector = *pair.first;
        const VectorConstView<float64>::const_iterator jointProbabilityIterator = jointProbabilityVector.cbegin();
        float64 sumOfJointProbabilities = pair.second;
        setArrayToZeros(probabilitiesBegin, numLabels);
        uint32 i = 0;

        for (auto it = labelVectorSet_.cbegin(); it != labelVectorSet_.cend(); it++) {
            const LabelVector& labelVector = *((*it).first);
            uint32 numRelevantLabels = labelVector.getNumElements();
            LabelVector::const_iterator labelIndexIterator = labelVector.cbegin();
            float64 jointProbability = jointProbabilityIterator[i];
            jointProbability = normalizeJointProbability(jointProbability, sumOfJointProbabilities);

            for (uint32 j = 0; j < numRelevantLabels; j++) {
                uint32 labelIndex = labelIndexIterator[j];
                probabilitiesBegin[labelIndex] += jointProbability;
            }

            i++;
        }
    }

}
