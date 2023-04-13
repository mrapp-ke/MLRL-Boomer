#include "boosting/prediction/transformation_binary_example_wise.hpp"

#include "common/iterator/binary_forward_iterator.hpp"

namespace boosting {

    static inline const LabelVector* measureDistance(VectorConstView<float64>::const_iterator scoresBegin,
                                                     VectorConstView<float64>::const_iterator scoresEnd,
                                                     LabelVectorSet::const_iterator labelVectorIterator,
                                                     const IDistanceMeasure& distanceMeasure, float64& distance,
                                                     uint32& count) {
        const auto& entry = *labelVectorIterator;
        const std::unique_ptr<LabelVector>& labelVectorPtr = entry.first;
        distance = distanceMeasure.measureDistance(*labelVectorPtr, scoresBegin, scoresEnd);
        count = entry.second;
        return labelVectorPtr.get();
    }

    static inline const LabelVector& findClosestLabelVector(VectorConstView<float64>::const_iterator scoresBegin,
                                                            VectorConstView<float64>::const_iterator scoresEnd,
                                                            const LabelVectorSet& labelVectorSet,
                                                            const IDistanceMeasure& distanceMeasure) {
        float64 minDistance;
        uint32 maxCount;
        LabelVectorSet::const_iterator labelVectorIterator = labelVectorSet.cbegin();
        const LabelVector* closestLabelVector =
          measureDistance(scoresBegin, scoresEnd, labelVectorIterator, distanceMeasure, minDistance, maxCount);
        labelVectorIterator++;

        for (; labelVectorIterator != labelVectorSet.cend(); labelVectorIterator++) {
            float64 distance;
            uint32 count;
            const LabelVector* labelVector =
              measureDistance(scoresBegin, scoresEnd, labelVectorIterator, distanceMeasure, distance, count);

            if (distance < minDistance || (distance == minDistance && count > maxCount)) {
                closestLabelVector = labelVector;
                minDistance = distance;
                maxCount = count;
            }
        }

        return *closestLabelVector;
    }

    ExampleWiseBinaryTransformation::ExampleWiseBinaryTransformation(
      const LabelVectorSet& labelVectorSet, std::unique_ptr<IDistanceMeasure> distanceMeasurePtr)
        : labelVectorSet_(labelVectorSet), distanceMeasurePtr_(std::move(distanceMeasurePtr)) {}

    void ExampleWiseBinaryTransformation::apply(VectorConstView<float64>::const_iterator scoresBegin,
                                                VectorConstView<float64>::const_iterator scoresEnd,
                                                VectorView<uint8>::iterator predictionBegin,
                                                VectorView<uint8>::iterator predictionEnd) const {
        const LabelVector& labelVector =
          findClosestLabelVector(scoresBegin, scoresEnd, labelVectorSet_, *distanceMeasurePtr_);
        uint32 numLabels = predictionEnd - predictionBegin;
        auto labelIterator = make_binary_forward_iterator(labelVector.cbegin(), labelVector.cend());

        for (uint32 i = 0; i < numLabels; i++) {
            bool label = *labelIterator;
            predictionBegin[i] = label ? 1 : 0;
            labelIterator++;
        }
    }

    void ExampleWiseBinaryTransformation::apply(VectorConstView<float64>::const_iterator scoresBegin,
                                                VectorConstView<float64>::const_iterator scoresEnd,
                                                BinaryLilMatrix::row predictionRow) const {
        const LabelVector& labelVector =
          findClosestLabelVector(scoresBegin, scoresEnd, labelVectorSet_, *distanceMeasurePtr_);
        uint32 numIndices = labelVector.getNumElements();
        LabelVector::const_iterator indexIterator = labelVector.cbegin();
        predictionRow.reserve(numIndices);

        for (uint32 i = 0; i < numIndices; i++) {
            uint32 labelIndex = indexIterator[i];
            predictionRow.emplace_back(labelIndex);
        }
    }

}
