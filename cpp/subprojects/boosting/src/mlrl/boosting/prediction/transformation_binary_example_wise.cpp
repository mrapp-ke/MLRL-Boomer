#include "mlrl/boosting/prediction/transformation_binary_example_wise.hpp"

#include "mlrl/common/iterator/iterator_forward_sparse_binary.hpp"

namespace boosting {

    ExampleWiseBinaryTransformation::ExampleWiseBinaryTransformation(
      const LabelVectorSet& labelVectorSet, std::unique_ptr<IDistanceMeasure> distanceMeasurePtr)
        : labelVectorSet_(labelVectorSet), distanceMeasurePtr_(std::move(distanceMeasurePtr)) {}

    void ExampleWiseBinaryTransformation::apply(View<float64>::const_iterator scoresBegin,
                                                View<float64>::const_iterator scoresEnd,
                                                View<uint8>::iterator predictionBegin,
                                                View<uint8>::iterator predictionEnd) const {
        const LabelVector& labelVector =
          distanceMeasurePtr_->getClosestLabelVector(labelVectorSet_, scoresBegin, scoresEnd);
        uint32 numLabels = predictionEnd - predictionBegin;
        auto labelIterator = createBinarySparseForwardIterator(labelVector.cbegin(), labelVector.cend());

        for (uint32 i = 0; i < numLabels; i++) {
            bool label = *labelIterator;
            predictionBegin[i] = label ? 1 : 0;
            labelIterator++;
        }
    }

    void ExampleWiseBinaryTransformation::apply(View<float64>::const_iterator scoresBegin,
                                                View<float64>::const_iterator scoresEnd,
                                                BinaryLilMatrix::row predictionRow) const {
        const LabelVector& labelVector =
          distanceMeasurePtr_->getClosestLabelVector(labelVectorSet_, scoresBegin, scoresEnd);
        uint32 numIndices = labelVector.getNumElements();
        LabelVector::const_iterator indexIterator = labelVector.cbegin();
        predictionRow.reserve(numIndices);

        for (uint32 i = 0; i < numIndices; i++) {
            uint32 labelIndex = indexIterator[i];
            predictionRow.emplace_back(labelIndex);
        }
    }

}
