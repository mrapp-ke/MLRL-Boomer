#include "boosting/prediction/transformation_probability_label_wise.hpp"

namespace boosting {

    LabelWiseProbabilityTransformation::LabelWiseProbabilityTransformation(
      std::unique_ptr<ILabelWiseProbabilityFunction> labelWiseProbabilityFunctionPtr)
        : labelWiseProbabilityFunctionPtr_(std::move(labelWiseProbabilityFunctionPtr)) {}

    void LabelWiseProbabilityTransformation::apply(VectorConstView<float64>::const_iterator scoresBegin,
                                                   VectorConstView<float64>::const_iterator scoresEnd,
                                                   VectorView<float64>::iterator probabilitiesBegin,
                                                   VectorView<float64>::iterator probabilitiesEnd) const {
        uint32 numScores = scoresEnd - scoresBegin;

        for (uint32 i = 0; i < numScores; i++) {
            float64 score = scoresBegin[i];
            float64 probability = labelWiseProbabilityFunctionPtr_->transform(score);
            probabilitiesBegin[i] = probability;
        }
    }

}
