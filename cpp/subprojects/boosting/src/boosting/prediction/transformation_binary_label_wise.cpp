#include "boosting/prediction/transformation_binary_label_wise.hpp"

namespace boosting {

    LabelWiseBinaryTransformation::LabelWiseBinaryTransformation(float64 threshold) : threshold_(threshold) {}

    void LabelWiseBinaryTransformation::apply(CContiguousConstView<float64>::value_const_iterator realBegin,
                                              CContiguousConstView<float64>::value_const_iterator realEnd,
                                              CContiguousView<uint8>::value_iterator predictionBegin,
                                              CContiguousView<uint8>::value_iterator predictionEnd) const {
        uint32 numPredictions = realEnd - realBegin;

        for (uint32 i = 0; i < numPredictions; i++) {
            float64 realPrediction = realBegin[i];
            uint8 binaryPrediction = realPrediction > threshold_ ? 1 : 0;
            predictionBegin[i] = binaryPrediction;
        }
    }

    void LabelWiseBinaryTransformation::apply(CContiguousConstView<float64>::value_const_iterator realBegin,
                                              CContiguousConstView<float64>::value_const_iterator realEnd,
                                              BinaryLilMatrix::row predictionRow) const {
        uint32 numPredictions = realEnd - realBegin;

        for (uint32 i = 0; i < numPredictions; i++) {
            float64 realPrediction = realBegin[i];

            if (realPrediction > threshold_) {
                predictionRow.emplace_back(i);
            }
        }
    }

    bool LabelWiseBinaryTransformation::shouldInitPredictionMatrix() const {
        return false;
    }

}
