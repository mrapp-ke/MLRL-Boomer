#include "mlrl/boosting/prediction/transformation_probability_marginalized.hpp"

namespace boosting {

    MarginalizedProbabilityTransformation::MarginalizedProbabilityTransformation(
      const LabelVectorSet& labelVectorSet, std::unique_ptr<IJointProbabilityFunction> jointProbabilityFunctionPtr)
        : labelVectorSet_(labelVectorSet), jointProbabilityFunctionPtr_(std::move(jointProbabilityFunctionPtr)) {}

    void MarginalizedProbabilityTransformation::apply(View<float64>::const_iterator scoresBegin,
                                                      View<float64>::const_iterator scoresEnd,
                                                      View<float64>::iterator probabilitiesBegin,
                                                      View<float64>::iterator probabilitiesEnd) const {
        std::unique_ptr<DenseVector<float64>> jointProbabilityVectorPtr =
          jointProbabilityFunctionPtr_->transformScoresIntoJointProbabilities(labelVectorSet_, scoresBegin, scoresEnd);
        DenseVector<float64>::const_iterator jointProbabilityIterator = jointProbabilityVectorPtr->cbegin();
        uint32 numLabels = probabilitiesEnd - probabilitiesBegin;
        setViewToZeros(probabilitiesBegin, numLabels);
        LabelVectorSet::const_iterator labelVectorIterator = labelVectorSet_.cbegin();
        uint32 numLabelVectors = labelVectorSet_.getNumLabelVectors();

        for (uint32 i = 0; i < numLabelVectors; i++) {
            const LabelVector& labelVector = *labelVectorIterator[i];
            uint32 numRelevantLabels = labelVector.getNumElements();
            LabelVector::const_iterator labelIndexIterator = labelVector.cbegin();
            float64 jointProbability = jointProbabilityIterator[i];

            for (uint32 j = 0; j < numRelevantLabels; j++) {
                uint32 labelIndex = labelIndexIterator[j];
                probabilitiesBegin[labelIndex] += jointProbability;
            }
        }
    }

}
