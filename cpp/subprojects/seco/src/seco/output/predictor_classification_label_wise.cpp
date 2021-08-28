#include "seco/output/predictor_classification_label_wise.hpp"
#include "common/model/head_complete.hpp"
#include "common/model/head_partial.hpp"
#include "common/validation.hpp"
#include "omp.h"


namespace seco {

    static inline void applyCompleteHead(const CompleteHead& head, CContiguousView<uint8>::iterator begin,
                                         CContiguousView<uint8>::iterator end, CContiguousView<uint8>::iterator mask) {
        CompleteHead::score_const_iterator iterator = head.scores_cbegin();
        uint32 numElements = head.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            if (mask[i] == 0) {
                uint8 prediction = iterator[i] > 0;
                begin[i] = prediction;
                mask[i] = 1;
            }
        }
    }

    static inline void applyPartialHead(const PartialHead& head, CContiguousView<uint8>::iterator begin,
                                        CContiguousView<uint8>::iterator end, CContiguousView<uint8>::iterator mask) {
        PartialHead::score_const_iterator scoreIterator = head.scores_cbegin();
        PartialHead::index_const_iterator indexIterator = head.indices_cbegin();
        uint32 numElements = head.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];

            if (mask[index] == 0) {
                uint8 prediction = scoreIterator[i] > 0;
                begin[index] = prediction;
                mask[index] = 1;
            }
        }
    }

    static inline void applyHead(const IHead& head, CContiguousView<uint8>& predictionMatrix,
                                 CContiguousView<uint8>::iterator mask, uint32 row) {
        auto completeHeadVisitor = [&, row](const CompleteHead& head) {
            applyCompleteHead(head, predictionMatrix.row_begin(row), predictionMatrix.row_end(row), mask);
        };
        auto partialHeadVisitor = [&, row](const PartialHead& head) {
            applyPartialHead(head, predictionMatrix.row_begin(row), predictionMatrix.row_end(row), mask);
        };
        head.visit(completeHeadVisitor, partialHeadVisitor);
    }

    // TODO Replace with iterator adaptor
    template<typename IndexIterator, typename ScoreIterator>
    static inline void advance(IndexIterator& indexIterator, ScoreIterator& scoreIterator, ScoreIterator scoresEnd) {
        for (; scoreIterator != scoresEnd; scoreIterator++) {
            float64 score = *scoreIterator;

            if (score > 0) {
                return;
            }

            indexIterator++;
        }
    }

    template<typename IndexIterator, typename ScoreIterator>
    static inline uint32 addFirst(IndexIterator& indexIterator, ScoreIterator& scoreIterator, ScoreIterator scoresEnd,
                                  LilMatrix<uint8>::Row& row) {
        advance(indexIterator, scoreIterator, scoresEnd);

        if (scoreIterator != scoresEnd) {
            uint32 labelIndex = *indexIterator;

            if (row.empty()) {
                row.emplace_front(labelIndex, 1);
                scoreIterator++;
                indexIterator++;
                advance(indexIterator, scoreIterator, scoresEnd);
                return 1;
            } else {
                LilMatrix<uint8>::Row::iterator it = row.begin();
                uint32 firstIndex = (*it).index;

                if (labelIndex == firstIndex) {
                    scoreIterator++;
                    indexIterator++;
                    advance(indexIterator, scoreIterator, scoresEnd);
                } else if (labelIndex < firstIndex) {
                    row.emplace_front(labelIndex, 1);
                    scoreIterator++;
                    indexIterator++;
                    advance(indexIterator, scoreIterator, scoresEnd);
                    return 1;
                }
            }
        }

        return 0;
    }

    template<typename IndexIterator, typename ValueIterator>
    static inline uint32 applyHead(IndexIterator indexIterator, ValueIterator scoreIterator, ValueIterator scoresEnd,
                                   LilMatrix<uint8>::Row& row) {
        uint32 numNonZeroElements = addFirst(indexIterator, scoreIterator, scoresEnd, row);
        LilMatrix<uint8>::Row::iterator prevIt = row.begin();
        LilMatrix<uint8>::Row::iterator it = prevIt;
        it++;

        while (scoreIterator != scoresEnd && it != row.end()) {
            uint32 labelIndex = *indexIterator;
            uint32 currentIndex = (*it).index;
            LilMatrix<uint8>::Row::iterator nextIt = it;
            nextIt++;

            while (labelIndex > currentIndex && nextIt != row.end()) {
                uint32 nextIndex = (*nextIt).index;

                if (labelIndex >= nextIndex) {
                    currentIndex = nextIndex;
                    prevIt = it;
                    it = nextIt;
                    nextIt++;
                } else {
                    break;
                }
            }

            if (labelIndex > currentIndex) {
                prevIt = row.emplace_after(it, labelIndex, 1);
                numNonZeroElements++;
                it = prevIt;
            } else if (labelIndex < currentIndex) {
                prevIt = row.emplace_after(prevIt, labelIndex, 1);
                numNonZeroElements++;
                it = prevIt;
            } else {
                prevIt = it;
            }

            scoreIterator++;
            indexIterator++;
            advance(indexIterator, scoreIterator, scoresEnd);
            it++;
        }

        while (scoreIterator != scoresEnd) {
            uint32 labelIndex = *indexIterator;
            prevIt = row.emplace_after(prevIt, labelIndex, 1);
            numNonZeroElements++;
            scoreIterator++;
            indexIterator++;
            advance(indexIterator, scoreIterator, scoresEnd);
        }

        return numNonZeroElements;
    }

    static inline uint32 applyHead(const IHead& head, LilMatrix<uint8>::Row& row) {
        uint32 numNonZeroElements;
        auto completeHeadVisitor = [&](const CompleteHead& head) mutable {
            numNonZeroElements = applyHead(IndexIterator(0), head.scores_cbegin(), head.scores_cend(), row);
        };
        auto partialHeadVisitor = [&](const PartialHead& head) mutable {
            numNonZeroElements = applyHead(head.indices_cbegin(), head.scores_cbegin(), head.scores_cend(), row);
        };
        head.visit(completeHeadVisitor, partialHeadVisitor);
        return numNonZeroElements;
    }

    LabelWiseClassificationPredictor::LabelWiseClassificationPredictor(uint32 numThreads)
        : numThreads_(numThreads) {
        assertGreaterOrEqual<uint32>("numThreads", numThreads, 1);
    }

    void LabelWiseClassificationPredictor::predict(const CContiguousFeatureMatrix& featureMatrix,
                                                   CContiguousView<uint8>& predictionMatrix,
                                                   const RuleModel& model, const LabelVectorSet* labelVectors) const {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numLabels = predictionMatrix.getNumCols();
        const CContiguousFeatureMatrix* featureMatrixPtr = &featureMatrix;
        CContiguousView<uint8>* predictionMatrixPtr = &predictionMatrix;
        const RuleModel* modelPtr = &model;

        #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(modelPtr) \
        firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) schedule(dynamic) num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            uint8 mask[numLabels] = {};

            for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                const Rule& rule = *it;
                const IBody& body = rule.getBody();

                if (body.covers(featureMatrixPtr->row_cbegin(i), featureMatrixPtr->row_cend(i))) {
                    const IHead& head = rule.getHead();
                    applyHead(head, *predictionMatrixPtr, &mask[0], i);
                }
            }
        }
    }

    void LabelWiseClassificationPredictor::predict(const CsrFeatureMatrix& featureMatrix,
                                                   CContiguousView<uint8>& predictionMatrix,
                                                   const RuleModel& model, const LabelVectorSet* labelVectors) const {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numFeatures = featureMatrix.getNumCols();
        uint32 numLabels = predictionMatrix.getNumCols();
        const CsrFeatureMatrix* featureMatrixPtr = &featureMatrix;
        CContiguousView<uint8>* predictionMatrixPtr = &predictionMatrix;
        const RuleModel* modelPtr = &model;

        #pragma omp parallel for firstprivate(numExamples) firstprivate(numFeatures) firstprivate(numLabels) \
        firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) schedule(dynamic) \
        num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            uint8 mask[numLabels] = {};
            float32 tmpArray1[numFeatures];
            uint32 tmpArray2[numFeatures] = {};
            uint32 n = 1;

            for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                const Rule& rule = *it;
                const IBody& body = rule.getBody();

                if (body.covers(featureMatrixPtr->row_indices_cbegin(i), featureMatrixPtr->row_indices_cend(i),
                                featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                                &tmpArray1[0], &tmpArray2[0], n)) {
                    const IHead& head = rule.getHead();
                    applyHead(head, *predictionMatrixPtr, &mask[0], i);
                }

                n++;
            }
        }
    }

    std::unique_ptr<SparsePredictionMatrix<uint8>> LabelWiseClassificationPredictor::predict(
            const CContiguousFeatureMatrix& featureMatrix, uint32 numLabels, const RuleModel& model,
            const LabelVectorSet* labelVectors) const {
        uint32 numExamples = featureMatrix.getNumRows();
        std::unique_ptr<LilMatrix<uint8>> lilMatrixPtr = std::make_unique<LilMatrix<uint8>>(numExamples);
        const CContiguousFeatureMatrix* featureMatrixPtr = &featureMatrix;
        LilMatrix<uint8>* predictionMatrixPtr = lilMatrixPtr.get();
        const RuleModel* modelPtr = &model;
        uint32 numNonZeroElements = 0;

        #pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) firstprivate(numLabels) \
        firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) schedule(dynamic) \
        num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            LilMatrix<uint8>::Row& row = predictionMatrixPtr->getRow(i);

            for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                const Rule& rule = *it;
                const IBody& body = rule.getBody();

                if (body.covers(featureMatrixPtr->row_cbegin(i), featureMatrixPtr->row_cend(i))) {
                    const IHead& head = rule.getHead();
                    numNonZeroElements += applyHead(head, row);
                }
            }
        }

        return std::make_unique<SparsePredictionMatrix<uint8>>(std::move(lilMatrixPtr), numLabels, numNonZeroElements);
    }

    std::unique_ptr<SparsePredictionMatrix<uint8>> LabelWiseClassificationPredictor::predict(
            const CsrFeatureMatrix& featureMatrix, uint32 numLabels, const RuleModel& model,
            const LabelVectorSet* labelVectors) const {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numFeatures = featureMatrix.getNumCols();
        std::unique_ptr<LilMatrix<uint8>> lilMatrixPtr = std::make_unique<LilMatrix<uint8>>(numExamples);
        const CsrFeatureMatrix* featureMatrixPtr = &featureMatrix;
        LilMatrix<uint8>* predictionMatrixPtr = lilMatrixPtr.get();
        const RuleModel* modelPtr = &model;
        uint32 numNonZeroElements = 0;

        #pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) firstprivate(numFeatures) \
        firstprivate(numLabels) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
        firstprivate(predictionMatrixPtr) schedule(dynamic) num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            LilMatrix<uint8>::Row& row = predictionMatrixPtr->getRow(i);
            float32 tmpArray1[numFeatures];
            uint32 tmpArray2[numFeatures] = {};
            uint32 n = 1;

            for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                const Rule& rule = *it;
                const IBody& body = rule.getBody();

                if (body.covers(featureMatrixPtr->row_indices_cbegin(i), featureMatrixPtr->row_indices_cend(i),
                                featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                                &tmpArray1[0], &tmpArray2[0], n)) {
                    const IHead& head = rule.getHead();
                    numNonZeroElements += applyHead(head, row);
                }

                n++;
            }
        }

        return std::make_unique<SparsePredictionMatrix<uint8>>(std::move(lilMatrixPtr), numLabels, numNonZeroElements);
    }

}
