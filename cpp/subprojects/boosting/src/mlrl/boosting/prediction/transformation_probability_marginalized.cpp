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
        auto jointProbabilityIterator = jointProbabilityVectorPtr->cbegin();
        std::fill(probabilitiesBegin, probabilitiesEnd, 0);
        auto labelVectorIterator = labelVectorSet_.cbegin();
        uint32 numLabelVectors = labelVectorSet_.getNumLabelVectors();

        for (uint32 i = 0; i < numLabelVectors; i++) {
            const LabelVector& labelVector = *labelVectorIterator[i];
            uint32 numRelevantLabels = labelVector.getNumElements();
            auto labelIndexIterator = labelVector.cbegin();
            float64 jointProbability = jointProbabilityIterator[i];

            for (uint32 j = 0; j < numRelevantLabels; j++) {
                uint32 labelIndex = labelIndexIterator[j];
                probabilitiesBegin[labelIndex] += jointProbability;
            }
        }
    }

}
