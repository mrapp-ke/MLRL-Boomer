#include "mlrl/boosting/prediction/transformation_probability_output_wise.hpp"

namespace boosting {

    OutputWiseProbabilityTransformation::OutputWiseProbabilityTransformation(
      std::unique_ptr<IMarginalProbabilityFunction> marginalProbabilityFunctionPtr)
        : marginalProbabilityFunctionPtr_(std::move(marginalProbabilityFunctionPtr)) {}

    void OutputWiseProbabilityTransformation::apply(View<float64>::const_iterator scoresBegin,
                                                    View<float64>::const_iterator scoresEnd,
                                                    View<float64>::iterator probabilitiesBegin,
                                                    View<float64>::iterator probabilitiesEnd) const {
        uint32 numScores = scoresEnd - scoresBegin;

        for (uint32 i = 0; i < numScores; i++) {
            float64 score = scoresBegin[i];
            float64 probability = marginalProbabilityFunctionPtr_->transformScoreIntoMarginalProbability(i, score);
            probabilitiesBegin[i] = probability;
        }
    }

}
