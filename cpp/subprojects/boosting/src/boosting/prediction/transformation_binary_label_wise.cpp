#include "boosting/prediction/transformation_binary_label_wise.hpp"

namespace boosting {

    LabelWiseBinaryTransformation::LabelWiseBinaryTransformation(float64 threshold) : threshold_(threshold) {}

    void LabelWiseBinaryTransformation::apply(VectorConstView<float64>::const_iterator scoresBegin,
                                              VectorConstView<float64>::const_iterator scoresEnd,
                                              VectorView<uint8>::iterator predictionBegin,
                                              VectorView<uint8>::iterator predictionEnd) const {
        uint32 numPredictions = scoresEnd - scoresBegin;

        for (uint32 i = 0; i < numPredictions; i++) {
            float64 score = scoresBegin[i];
            uint8 binaryPrediction = score > threshold_ ? 1 : 0;
            predictionBegin[i] = binaryPrediction;
        }
    }

    void LabelWiseBinaryTransformation::apply(VectorConstView<float64>::const_iterator scoresBegin,
                                              VectorConstView<float64>::const_iterator scoresEnd,
                                              BinaryLilMatrix::row predictionRow) const {
        uint32 numPredictions = scoresEnd - scoresBegin;

        for (uint32 i = 0; i < numPredictions; i++) {
            float64 score = scoresBegin[i];

            if (score > threshold_) {
                predictionRow.emplace_back(i);
            }
        }
    }

}
