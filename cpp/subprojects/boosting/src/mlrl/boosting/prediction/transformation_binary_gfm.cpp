#include "mlrl/boosting/prediction/transformation_binary_gfm.hpp"

#include "mlrl/common/data/matrix_sparse_set.hpp"
#include "mlrl/common/data/vector_sparse_array.hpp"

#include <algorithm>

namespace boosting {

    static inline uint32 getMaxLabelCardinality(const LabelVectorSet& labelVectorSet) {
        LabelVectorSet::const_iterator labelVectorIterator = labelVectorSet.cbegin();
        uint32 numLabelVectors = labelVectorSet.getNumLabelVectors();
        uint32 maxLabelCardinality = 0;

        for (uint32 i = 0; i < numLabelVectors; i++) {
            const LabelVector& labelVector = *labelVectorIterator[i];
            uint32 numRelevantLabels = labelVector.getNumElements();

            if (numRelevantLabels > maxLabelCardinality) {
                maxLabelCardinality = numRelevantLabels;
            }
        }

        return maxLabelCardinality;
    }

    static inline float64 calculateMarginalizedProbabilities(SparseSetView<float64>& probabilities, uint32 numLabels,
                                                             View<float64>::const_iterator jointProbabilityIterator,
                                                             const LabelVectorSet& labelVectorSet) {
        LabelVectorSet::const_iterator labelVectorIterator = labelVectorSet.cbegin();
        uint32 numLabelVectors = labelVectorSet.getNumLabelVectors();
        float64 nullVectorProbability = 0;

        for (uint32 i = 0; i < numLabelVectors; i++) {
            const LabelVector& labelVector = *labelVectorIterator[i];
            uint32 numRelevantLabels = labelVector.getNumElements();
            float64 jointProbability = jointProbabilityIterator[i];

            if (numRelevantLabels > 0) {
                LabelVector::const_iterator labelIndexIterator = labelVector.cbegin();

                for (uint32 j = 0; j < numRelevantLabels; j++) {
                    uint32 labelIndex = labelIndexIterator[j];
                    SparseSetView<float64>::row row = probabilities[labelIndex];
                    IndexedValue<float64>& indexedValue = row.emplace(numRelevantLabels - 1, 0.0);
                    indexedValue.value += jointProbability;
                }
            } else {
                nullVectorProbability = jointProbability;
            }
        }

        return nullVectorProbability;
    }

    static inline float64 createAndEvaluateLabelVector(ResizableSparseArrayVector<float64>::iterator iterator,
                                                       uint32 numLabels, const SparseSetView<float64>& probabilities,
                                                       uint32 k) {
        for (uint32 i = 0; i < numLabels; i++) {
            float64 weightedProbability = 0;

            for (auto it = probabilities.values_cbegin(i); it != probabilities.values_cend(i); it++) {
                const IndexedValue<float64>& indexedValue = *it;
                weightedProbability += (2 * indexedValue.value) / (float64) (indexedValue.index + k + 1);
            }

            IndexedValue<float64>& entry = iterator[i];
            entry.index = i;
            entry.value = weightedProbability;
        }

        std::partial_sort(iterator, &iterator[k], &iterator[numLabels],
                          [=](const IndexedValue<float64>& a, const IndexedValue<float64>& b) {
            return a.value > b.value;
        });

        float64 quality = 0;

        for (uint32 i = 0; i < k; i++) {
            quality += iterator[i].value;
        }

        return quality;
    }

    static inline void storePrediction(const ResizableSparseArrayVector<float64>& tmpVector,
                                       View<uint8>::iterator predictionIterator, uint32 numLabels) {
        util::setViewToZeros(predictionIterator, numLabels);
        uint32 numRelevantLabels = tmpVector.getNumElements();
        ResizableSparseArrayVector<float64>::const_iterator iterator = tmpVector.cbegin();

        for (uint32 i = 0; i < numRelevantLabels; i++) {
            uint32 labelIndex = iterator[i].index;
            predictionIterator[labelIndex] = 1;
        }
    }

    static inline void storePrediction(ResizableSparseArrayVector<float64>& tmpVector,
                                       BinaryLilMatrix::row predictionRow, uint32 numLabels) {
        uint32 numRelevantLabels = tmpVector.getNumElements();

        if (numRelevantLabels > 0) {
            ResizableSparseArrayVector<float64>::iterator iterator = tmpVector.begin();
            std::sort(iterator, tmpVector.end(), IndexedValue<float64>::CompareIndex());
            predictionRow.reserve(numRelevantLabels);

            for (uint32 i = 0; i < numRelevantLabels; i++) {
                predictionRow.emplace_back(iterator[i].index);
            }
        }
    }

    template<typename Prediction>
    static inline void predictGfm(View<float64>::const_iterator scoresBegin, View<float64>::const_iterator scoresEnd,
                                  Prediction prediction, const IJointProbabilityFunction& jointProbabilityFunction,
                                  const LabelVectorSet& labelVectorSet, uint32 maxLabelCardinality) {
        std::unique_ptr<DenseVector<float64>> jointProbabilityVectorPtr =
          jointProbabilityFunction.transformScoresIntoJointProbabilities(labelVectorSet, scoresBegin, scoresEnd);
        DenseVector<float64>::const_iterator jointProbabilityIterator = jointProbabilityVectorPtr->cbegin();
        uint32 numLabels = scoresEnd - scoresBegin;
        SparseSetMatrix<float64> marginalizedProbabilities(numLabels, maxLabelCardinality);
        float64 bestQuality = calculateMarginalizedProbabilities(marginalizedProbabilities.getView(), numLabels,
                                                                 jointProbabilityIterator, labelVectorSet);
        ResizableSparseArrayVector<float64> tmpVector1(numLabels);
        tmpVector1.setNumElements(0, false);
        ResizableSparseArrayVector<float64> tmpVector2(numLabels);
        ResizableSparseArrayVector<float64>* bestVectorPtr = &tmpVector1;
        ResizableSparseArrayVector<float64>* tmpVectorPtr = &tmpVector2;

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 k = i + 1;
            float64 quality =
              createAndEvaluateLabelVector(tmpVectorPtr->begin(), numLabels, marginalizedProbabilities.getView(), k);

            if (quality > bestQuality) {
                bestQuality = quality;
                tmpVectorPtr->setNumElements(k, false);
                ResizableSparseArrayVector<float64>* tmpPtr = bestVectorPtr;
                bestVectorPtr = tmpVectorPtr;
                tmpVectorPtr = tmpPtr;
            }
        }

        storePrediction(*bestVectorPtr, prediction, numLabels);
    }

    GfmBinaryTransformation::GfmBinaryTransformation(
      const LabelVectorSet& labelVectorSet, std::unique_ptr<IJointProbabilityFunction> jointProbabilityFunctionPtr)
        : labelVectorSet_(labelVectorSet), maxLabelCardinality_(getMaxLabelCardinality(labelVectorSet)),
          jointProbabilityFunctionPtr_(std::move(jointProbabilityFunctionPtr)) {}

    void GfmBinaryTransformation::apply(View<float64>::const_iterator scoresBegin,
                                        View<float64>::const_iterator scoresEnd, View<uint8>::iterator predictionBegin,
                                        View<uint8>::iterator predictionEnd) const {
        predictGfm(scoresBegin, scoresEnd, predictionBegin, *jointProbabilityFunctionPtr_, labelVectorSet_,
                   maxLabelCardinality_);
    }

    void GfmBinaryTransformation::apply(View<float64>::const_iterator scoresBegin,
                                        View<float64>::const_iterator scoresEnd,
                                        BinaryLilMatrix::row predictionRow) const {
        predictGfm<BinaryLilMatrix::row>(scoresBegin, scoresEnd, predictionRow, *jointProbabilityFunctionPtr_,
                                         labelVectorSet_, maxLabelCardinality_);
    }

}
