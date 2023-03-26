#include "boosting/prediction/transformation_binary_label_wise.hpp"

namespace boosting {

    LabelWiseBinaryTransformation::LabelWiseBinaryTransformation(float64 threshold) : threshold_(threshold) {}

    void LabelWiseBinaryTransformation::apply(VectorConstView<float64>::const_iterator realBegin,
                                              VectorConstView<float64>::const_iterator realEnd,
                                              VectorView<uint8>::iterator predictionBegin,
                                              VectorView<uint8>::iterator predictionEnd) const {
        uint32 numPredictions = realEnd - realBegin;

        for (uint32 i = 0; i < numPredictions; i++) {
            float64 realPrediction = realBegin[i];
            uint8 binaryPrediction = realPrediction > threshold_ ? 1 : 0;
            predictionBegin[i] = binaryPrediction;
        }
    }

    void LabelWiseBinaryTransformation::apply(VectorConstView<float64>::const_iterator realBegin,
                                              VectorConstView<float64>::const_iterator realEnd,
                                              BinaryLilMatrix::row predictionRow) const {
        uint32 numPredictions = realEnd - realBegin;

        for (uint32 i = 0; i < numPredictions; i++) {
            float64 realPrediction = realBegin[i];

            if (realPrediction > threshold_) {
                predictionRow.emplace_back(i);
            }
        }
    }

}
