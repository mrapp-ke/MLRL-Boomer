#include "seco/output/predictor_classification_label_wise.hpp"
#include "common/data/vector_bit.hpp"
#include "common/iterator/index_iterator.hpp"
#include "common/iterator/non_zero_index_forward_iterator.hpp"
#include "common/model/head_complete.hpp"
#include "common/model/head_partial.hpp"
#include "common/validation.hpp"
#include "omp.h"


namespace seco {

    static inline void applyCompleteHead(const CompleteHead& head, CContiguousView<uint8>::iterator begin,
                                         CContiguousView<uint8>::iterator end, BitVector& mask) {
        CompleteHead::score_const_iterator iterator = head.scores_cbegin();
        uint32 numElements = head.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            if (!mask[i]) {
                uint8 prediction = iterator[i] > 0;
                begin[i] = prediction;
                mask.set(i, true);
            }
        }
    }

    static inline void applyPartialHead(const PartialHead& head, CContiguousView<uint8>::iterator begin,
                                        CContiguousView<uint8>::iterator end, BitVector& mask) {
        PartialHead::score_const_iterator scoreIterator = head.scores_cbegin();
        PartialHead::index_const_iterator indexIterator = head.indices_cbegin();
        uint32 numElements = head.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];

            if (!mask[index]) {
                uint8 prediction = scoreIterator[i] > 0;
                begin[index] = prediction;
                mask.set(index, true);
            }
        }
    }

    static inline void applyHead(const IHead& head, CContiguousView<uint8>& predictionMatrix, BitVector& mask,
                                 uint32 row) {
        auto completeHeadVisitor = [&, row](const CompleteHead& head) {
            applyCompleteHead(head, predictionMatrix.row_begin(row), predictionMatrix.row_end(row), mask);
        };
        auto partialHeadVisitor = [&, row](const PartialHead& head) {
            applyPartialHead(head, predictionMatrix.row_begin(row), predictionMatrix.row_end(row), mask);
        };
        head.visit(completeHeadVisitor, partialHeadVisitor);
    }

    template<typename ScoreIterator, typename IndexIterator>
    static inline uint32 addFirst(ScoreIterator& scoresBegin, ScoreIterator scoresEnd, IndexIterator indexIterator,
                                  BinaryLilMatrix::Row& row) {
        if (scoresBegin != scoresEnd) {
            uint32 index = indexIterator[*scoresBegin];

            if (row.empty()) {
                row.emplace_front(index);
                scoresBegin++;
                return 1;
            } else {
                BinaryLilMatrix::Row::iterator it = row.begin();
                uint32 firstIndex = *it;

                if (index == firstIndex) {
                    scoresBegin++;
                } else if (index < firstIndex) {
                    row.emplace_front(index);
                    scoresBegin++;
                    return 1;
                }
            }
        }

        return 0;
    }

    template<typename ScoreIterator, typename IndexIterator>
    static inline uint32 applyHead(ScoreIterator scoresBegin, ScoreIterator scoresEnd, IndexIterator indexIterator,
                                   BinaryLilMatrix::Row& row) {
        uint32 numNonZeroElements = addFirst(scoresBegin, scoresEnd, indexIterator, row);
        BinaryLilMatrix::Row::iterator prevIt = row.begin();
        BinaryLilMatrix::Row::iterator it = prevIt;
        it++;

        for (; scoresBegin != scoresEnd && it != row.end(); scoresBegin++) {
            uint32 index = indexIterator[*scoresBegin];
            uint32 currentIndex = *it;
            BinaryLilMatrix::Row::iterator nextIt = it;
            nextIt++;

            while (index > currentIndex && nextIt != row.end()) {
                uint32 nextIndex = *nextIt;

                if (index >= nextIndex) {
                    currentIndex = nextIndex;
                    prevIt = it;
                    it = nextIt;
                    nextIt++;
                } else {
                    break;
                }
            }

            if (index > currentIndex) {
                prevIt = row.emplace_after(it, index);
                numNonZeroElements++;
                it = prevIt;
            } else if (index < currentIndex) {
                prevIt = row.emplace_after(prevIt, index);
                numNonZeroElements++;
                it = prevIt;
            } else {
                prevIt = it;
            }

            it++;
        }

        for (; scoresBegin != scoresEnd; scoresBegin++) {
            uint32 index = indexIterator[*scoresBegin];
            prevIt = row.emplace_after(prevIt, index);
            numNonZeroElements++;
        }

        return numNonZeroElements;
    }

    static inline uint32 applyHead(const IHead& head, BinaryLilMatrix::Row& row) {
        uint32 numNonZeroElements;
        auto completeHeadVisitor = [&](const CompleteHead& head) mutable {
            numNonZeroElements = applyHead(
                make_non_zero_index_forward_iterator(head.scores_cbegin(), head.scores_cend()),
                make_non_zero_index_forward_iterator(head.scores_cend(), head.scores_cend()), IndexIterator(0), row);
        };
        auto partialHeadVisitor = [&](const PartialHead& head) mutable {
            numNonZeroElements = applyHead(
                make_non_zero_index_forward_iterator(head.scores_cbegin(), head.scores_cend()),
                make_non_zero_index_forward_iterator(head.scores_cend(), head.scores_cend()), head.indices_cbegin(),
                row);
        };
        head.visit(completeHeadVisitor, partialHeadVisitor);
        return numNonZeroElements;
    }

    /**
     * An implementation of the type `IClassificationPredictor` that allows to predict whether individual labels of
     * given query examples are relevant or irrelevant by processing rules of an existing rule-based model in the order
     * they have been learned. If a rule covers an example, its prediction (1 if the label is relevant, 0 otherwise) is
     * applied to each label individually, if none of the previous rules has already predicted for a particular example
     * and label.
     *
     * @tparam Model The type of the rule-based model that is used to obtain predictions
     */
    template<typename Model>
    class LabelWiseClassificationPredictor final : public IClassificationPredictor {

        private:

            const Model& model_;

            uint32 numThreads_;

        public:

            /**
             * @param model         A reference to an object of template type `Model` that should be used to obtain
             *                      predictions
             * @param numThreads    The number of CPU threads to be used to make predictions for different query
             *                      examples in parallel. Must be at least 1
             */
            LabelWiseClassificationPredictor(const Model& model, uint32 numThreads)
                : model_(model), numThreads_(numThreads) {

            }

            void predict(const CContiguousFeatureMatrix& featureMatrix,
                         CContiguousView<uint8>& predictionMatrix) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                uint32 numLabels = predictionMatrix.getNumCols();
                const CContiguousFeatureMatrix* featureMatrixPtr = &featureMatrix;
                CContiguousView<uint8>* predictionMatrixPtr = &predictionMatrix;
                const Model* modelPtr = &model_;

                #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(modelPtr) \
                firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) schedule(dynamic) \
                num_threads(numThreads_)
                for (int64 i = 0; i < numExamples; i++) {
                    BitVector mask(numLabels, true);

                    for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                        const RuleList::Rule& rule = *it;
                        const IBody& body = rule.getBody();

                        if (body.covers(featureMatrixPtr->row_cbegin(i), featureMatrixPtr->row_cend(i))) {
                            const IHead& head = rule.getHead();
                            applyHead(head, *predictionMatrixPtr, mask, i);
                        }
                    }
                }
            }

            void predict(const CsrFeatureMatrix& featureMatrix,
                         CContiguousView<uint8>& predictionMatrix) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                uint32 numFeatures = featureMatrix.getNumCols();
                uint32 numLabels = predictionMatrix.getNumCols();
                const CsrFeatureMatrix* featureMatrixPtr = &featureMatrix;
                CContiguousView<uint8>* predictionMatrixPtr = &predictionMatrix;
                const Model* modelPtr = &model_;

                #pragma omp parallel for firstprivate(numExamples) firstprivate(numFeatures) firstprivate(numLabels) \
                firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) \
                schedule(dynamic) num_threads(numThreads_)
                for (int64 i = 0; i < numExamples; i++) {
                    BitVector mask(numLabels, true);
                    float32* tmpArray1 = new float32[numFeatures];
                    uint32* tmpArray2 = new uint32[numFeatures] {};
                    uint32 n = 1;

                    for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                        const RuleList::Rule& rule = *it;
                        const IBody& body = rule.getBody();

                        if (body.covers(featureMatrixPtr->row_indices_cbegin(i), featureMatrixPtr->row_indices_cend(i),
                                        featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                                        &tmpArray1[0], &tmpArray2[0], n)) {
                            const IHead& head = rule.getHead();
                            applyHead(head, *predictionMatrixPtr, mask, i);
                        }

                        n++;
                    }

                    delete[] tmpArray1;
                    delete[] tmpArray2;
                }
            }

            std::unique_ptr<BinarySparsePredictionMatrix> predictSparse(const CContiguousFeatureMatrix& featureMatrix,
                                                                        uint32 numLabels) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                BinaryLilMatrix lilMatrix(numExamples);
                const CContiguousFeatureMatrix* featureMatrixPtr = &featureMatrix;
                BinaryLilMatrix* predictionMatrixPtr = &lilMatrix;
                const Model* modelPtr = &model_;
                uint32 numNonZeroElements = 0;

                #pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) \
                firstprivate(numLabels) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
                firstprivate(predictionMatrixPtr) schedule(dynamic) num_threads(numThreads_)
                for (int64 i = 0; i < numExamples; i++) {
                    BinaryLilMatrix::Row& row = predictionMatrixPtr->getRow(i);

                    for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                        const RuleList::Rule& rule = *it;
                        const IBody& body = rule.getBody();

                        if (body.covers(featureMatrixPtr->row_cbegin(i), featureMatrixPtr->row_cend(i))) {
                            const IHead& head = rule.getHead();
                            numNonZeroElements += applyHead(head, row);
                        }
                    }
                }

                return createBinarySparsePredictionMatrix(lilMatrix, numLabels, numNonZeroElements);
            }

            std::unique_ptr<BinarySparsePredictionMatrix> predictSparse(const CsrFeatureMatrix& featureMatrix,
                                                                        uint32 numLabels) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                uint32 numFeatures = featureMatrix.getNumCols();
                BinaryLilMatrix lilMatrix(numExamples);
                const CsrFeatureMatrix* featureMatrixPtr = &featureMatrix;
                BinaryLilMatrix* predictionMatrixPtr = &lilMatrix;
                const Model* modelPtr = &model_;
                uint32 numNonZeroElements = 0;

                #pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) \
                firstprivate(numFeatures) firstprivate(numLabels) firstprivate(modelPtr) \
                firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) schedule(dynamic) \
                num_threads(numThreads_)
                for (int64 i = 0; i < numExamples; i++) {
                    BinaryLilMatrix::Row& row = predictionMatrixPtr->getRow(i);
                    float32* tmpArray1 = new float32[numFeatures];
                    uint32* tmpArray2 = new uint32[numFeatures] {};
                    uint32 n = 1;

                    for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                        const RuleList::Rule& rule = *it;
                        const IBody& body = rule.getBody();

                        if (body.covers(featureMatrixPtr->row_indices_cbegin(i), featureMatrixPtr->row_indices_cend(i),
                                        featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                                        &tmpArray1[0], &tmpArray2[0], n)) {
                            const IHead& head = rule.getHead();
                            numNonZeroElements += applyHead(head, row);
                        }

                        n++;
                    }

                    delete[] tmpArray1;
                    delete[] tmpArray2;
                }

                return createBinarySparsePredictionMatrix(lilMatrix, numLabels, numNonZeroElements);
            }

    };

    LabelWiseClassificationPredictorFactory::LabelWiseClassificationPredictorFactory(uint32 numThreads)
        : numThreads_(numThreads) {
        assertGreaterOrEqual<uint32>("numThreads", numThreads, 1);
    }

    std::unique_ptr<IClassificationPredictor> LabelWiseClassificationPredictorFactory::create(
            const RuleList& model, const LabelVectorSet* labelVectorSet) const {
        return std::make_unique<LabelWiseClassificationPredictor<RuleList>>(model, numThreads_);
    }

}
