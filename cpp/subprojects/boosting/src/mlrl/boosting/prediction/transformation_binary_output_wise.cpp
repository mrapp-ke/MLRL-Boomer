#include "mlrl/boosting/prediction/transformation_binary_output_wise.hpp"

namespace boosting {

    OutputWiseBinaryTransformation::OutputWiseBinaryTransformation(
      std::unique_ptr<IDiscretizationFunction> discretizationFunctionPtr)
        : discretizationFunctionPtr_(std::move(discretizationFunctionPtr)) {}

    void OutputWiseBinaryTransformation::apply(View<float64>::const_iterator scoresBegin,
                                               View<float64>::const_iterator scoresEnd,
                                               View<uint8>::iterator predictionBegin,
                                               View<uint8>::iterator predictionEnd) const {
        uint32 numPredictions = scoresEnd - scoresBegin;

        for (uint32 i = 0; i < numPredictions; i++) {
            float64 score = scoresBegin[i];
            uint8 binaryPrediction = discretizationFunctionPtr_->discretizeScore(i, score) ? 1 : 0;
            predictionBegin[i] = binaryPrediction;
        }
    }

    void OutputWiseBinaryTransformation::apply(View<float64>::const_iterator scoresBegin,
                                               View<float64>::const_iterator scoresEnd,
                                               BinaryLilMatrix::row predictionRow) const {
        uint32 numPredictions = scoresEnd - scoresBegin;

        for (uint32 i = 0; i < numPredictions; i++) {
            float64 score = scoresBegin[i];

            if (discretizationFunctionPtr_->discretizeScore(i, score)) {
                predictionRow.emplace_back(i);
            }
        }
    }

}
