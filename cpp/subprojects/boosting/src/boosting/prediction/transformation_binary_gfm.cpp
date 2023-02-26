#include "boosting/prediction/transformation_binary_gfm.hpp"

#include "common/data/matrix_sparse_set.hpp"
#include "common/data/vector_sparse_array.hpp"
#include "joint_probabilities.hpp"

#include <algorithm>

namespace boosting {

    static inline uint32 getMaxLabelCardinality(const LabelVectorSet& labelVectorSet) {
        uint32 maxLabelCardinality = 0;

        for (auto it = labelVectorSet.cbegin(); it != labelVectorSet.cend(); it++) {
            const auto& entry = *it;
            const std::unique_ptr<LabelVector>& labelVectorPtr = entry.first;
            uint32 numRelevantLabels = labelVectorPtr->getNumElements();

            if (numRelevantLabels > maxLabelCardinality) {
                maxLabelCardinality = numRelevantLabels;
            }
        }

        return maxLabelCardinality;
    }

    static inline float64 calculateMarginalizedProbabilities(
      SparseSetMatrix<float64>& probabilities, uint32 numLabels,
      VectorConstView<float64>::const_iterator jointProbabilityIterator, float64 sumOfJointProbabilities,
      const LabelVectorSet& labelVectorSet) {
        float64 nullVectorProbability = 0;
        uint32 i = 0;

        for (auto it = labelVectorSet.cbegin(); it != labelVectorSet.cend(); it++) {
            const auto& entry = *it;
            const std::unique_ptr<LabelVector>& labelVectorPtr = entry.first;
            uint32 numRelevantLabels = labelVectorPtr->getNumElements();
            float64 jointProbability = jointProbabilityIterator[i];
            jointProbability = normalizeJointProbability(jointProbability, sumOfJointProbabilities);

            if (numRelevantLabels > 0) {
                LabelVector::const_iterator labelIndexIterator = labelVectorPtr->cbegin();

                for (uint32 j = 0; j < numRelevantLabels; j++) {
                    uint32 labelIndex = labelIndexIterator[j];
                    SparseSetMatrix<float64>::row row = probabilities[labelIndex];
                    IndexedValue<float64>& indexedValue = row.emplace(numRelevantLabels - 1, 0.0);
                    indexedValue.value += jointProbability;
                }
            } else {
                nullVectorProbability = jointProbability;
            }

            i++;
        }

        return nullVectorProbability;
    }

    static inline float64 createAndEvaluateLabelVector(SparseArrayVector<float64>::iterator iterator, uint32 numLabels,
                                                       const SparseSetMatrix<float64>& probabilities, uint32 k) {
        for (uint32 i = 0; i < numLabels; i++) {
            float64 weightedProbability = 0;

            for (auto it = probabilities.row_cbegin(i); it != probabilities.row_cend(i); it++) {
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

    static inline void storePrediction(const SparseArrayVector<float64>& tmpVector,
                                       CContiguousView<uint8>::value_iterator predictionIterator) {
        uint32 numRelevantLabels = tmpVector.getNumElements();
        SparseArrayVector<float64>::const_iterator iterator = tmpVector.cbegin();

        for (uint32 i = 0; i < numRelevantLabels; i++) {
            uint32 labelIndex = iterator[i].index;
            predictionIterator[labelIndex] = 1;
        }
    }

    static inline void storePrediction(SparseArrayVector<float64>& tmpVector, BinaryLilMatrix::row predictionRow) {
        uint32 numRelevantLabels = tmpVector.getNumElements();

        if (numRelevantLabels > 0) {
            SparseArrayVector<float64>::iterator iterator = tmpVector.begin();
            std::sort(iterator, tmpVector.end(), [=](const IndexedValue<float64>& a, const IndexedValue<float64>& b) {
                return a.index < b.index;
            });

            for (uint32 i = 0; i < numRelevantLabels; i++) {
                predictionRow.emplace_back(iterator[i].index);
            }
        }
    }

    template<typename Prediction>
    static inline void predictGfm(CContiguousConstView<float64>::value_const_iterator scoresBegin,
                                  Prediction prediction, uint32 numLabels,
                                  const IProbabilityFunction& probabilityFunction, const LabelVectorSet& labelVectorSet,
                                  uint32 maxLabelCardinality) {
        std::pair<std::unique_ptr<DenseVector<float64>>, float64> pair =
          calculateJointProbabilities(scoresBegin, numLabels, labelVectorSet, probabilityFunction);
        const VectorConstView<float64>& jointProbabilityVector = *pair.first;
        VectorConstView<float64>::const_iterator jointProbabilityIterator = jointProbabilityVector.cbegin();
        float64 sumOfJointProbabilities = pair.second;
        SparseSetMatrix<float64> marginalProbabilities(numLabels, maxLabelCardinality);
        float64 bestQuality = calculateMarginalizedProbabilities(
          marginalProbabilities, numLabels, jointProbabilityIterator, sumOfJointProbabilities, labelVectorSet);

        SparseArrayVector<float64> tmpVector1(numLabels);
        tmpVector1.setNumElements(0, false);
        SparseArrayVector<float64> tmpVector2(numLabels);
        SparseArrayVector<float64>* bestVectorPtr = &tmpVector1;
        SparseArrayVector<float64>* tmpVectorPtr = &tmpVector2;

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 k = i + 1;
            float64 quality = createAndEvaluateLabelVector(tmpVectorPtr->begin(), numLabels, marginalProbabilities, k);

            if (quality > bestQuality) {
                bestQuality = quality;
                tmpVectorPtr->setNumElements(k, false);
                SparseArrayVector<float64>* tmpPtr = bestVectorPtr;
                bestVectorPtr = tmpVectorPtr;
                tmpVectorPtr = tmpPtr;
            }
        }

        storePrediction(*bestVectorPtr, prediction);
    }

    GfmBinaryTransformation::GfmBinaryTransformation(const LabelVectorSet& labelVectorSet,
                                                     std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr)
        : labelVectorSet_(labelVectorSet), maxLabelCardinality_(getMaxLabelCardinality(labelVectorSet)),
          probabilityFunctionPtr_(std::move(probabilityFunctionPtr)) {}

    void GfmBinaryTransformation::apply(CContiguousConstView<float64>::value_const_iterator realBegin,
                                        CContiguousConstView<float64>::value_const_iterator realEnd,
                                        CContiguousView<uint8>::value_iterator predictionBegin,
                                        CContiguousView<uint8>::value_iterator predictionEnd) const {
        uint32 numLabels = realEnd - realBegin;
        predictGfm(realBegin, predictionBegin, numLabels, *probabilityFunctionPtr_, labelVectorSet_,
                   maxLabelCardinality_);
    }

    void GfmBinaryTransformation::apply(CContiguousConstView<float64>::value_const_iterator realBegin,
                                        CContiguousConstView<float64>::value_const_iterator realEnd,
                                        BinaryLilMatrix::row predictionRow) const {
        uint32 numLabels = realEnd - realBegin;
        predictGfm<BinaryLilMatrix::row>(realBegin, predictionRow, numLabels, *probabilityFunctionPtr_, labelVectorSet_,
                                         maxLabelCardinality_);
    }

}
