#include "boosting/prediction/transformation_probability_label_wise.hpp"

namespace boosting {

    LabelWiseProbabilityTransformation::LabelWiseProbabilityTransformation(
      std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr)
        : probabilityFunctionPtr_(std::move(probabilityFunctionPtr)) {}

    void LabelWiseProbabilityTransformation::apply(CContiguousView<float64>::value_iterator scoresBegin,
                                                   CContiguousView<float64>::value_iterator scoresEnd) const {
        uint32 numScores = scoresEnd - scoresBegin;

        for (uint32 i = 0; i < numScores; i++) {
            float64 score = scoresBegin[i];
            float64 probability = probabilityFunctionPtr_->transform(score);
            scoresBegin[i] = probability;
        }
    }

}
