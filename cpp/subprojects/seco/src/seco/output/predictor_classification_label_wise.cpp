#include "seco/output/predictor_classification_label_wise.hpp"
#include "common/data/vector_bit.hpp"
#include "common/iterator/index_iterator.hpp"
#include "common/iterator/non_zero_index_forward_iterator.hpp"
#include "common/model/head_complete.hpp"
#include "common/model/head_partial.hpp"
#include "omp.h"


namespace seco {

    static inline void applyCompleteHead(const CompleteHead& head, CContiguousView<uint8>::value_iterator begin,
                                         CContiguousView<uint8>::value_iterator end, BitVector& mask) {
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

    static inline void applyPartialHead(const PartialHead& head, CContiguousView<uint8>::value_iterator begin,
                                        CContiguousView<uint8>::value_iterator end, BitVector& mask) {
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
            applyCompleteHead(head, predictionMatrix.row_values_begin(row), predictionMatrix.row_values_end(row), mask);
        };
        auto partialHeadVisitor = [&, row](const PartialHead& head) {
            applyPartialHead(head, predictionMatrix.row_values_begin(row), predictionMatrix.row_values_end(row), mask);
        };
        head.visit(completeHeadVisitor, partialHeadVisitor);
    }

    template<typename ScoreIterator, typename IndexIterator>
    static inline uint32 applyHead(ScoreIterator scoresBegin, ScoreIterator scoresEnd, IndexIterator indexIterator,
                                   BinaryLilMatrix::Row& row, uint32 numLabels) {
        if (scoresBegin != scoresEnd) {
            uint32 numElements = row.size();

            if (numElements > 0) {
                BinaryLilMatrix::Row::iterator end = row.end();
                BinaryLilMatrix::Row::iterator start = std::lower_bound(row.begin(), end, indexIterator[*scoresBegin]);
                uint32 bufferSize = end - start;
                uint32* buffer = new uint32[bufferSize];

                for (uint32 i = 0; i < bufferSize; i++) {
                    buffer[i] = start[i];
                }

                uint32 i = 0;

                for (uint32 n = 0; n < bufferSize; n++) {
                    uint32 index1 = i < bufferSize ? buffer[i] : numLabels;
                    uint32 index2 = scoresBegin != scoresEnd ? indexIterator[*scoresBegin] : numLabels;

                    if (index1 < index2) {
                        start[n] = index1;
                        i++;
                    } else if (index1 == index2) {
                        start[n] = index1;
                        i++;
                        scoresBegin++;
                    } else {
                        start[n] = index2;
                        scoresBegin++;
                    }
                }

                while (i < bufferSize) {
                    uint32 index1 = buffer[i];
                    uint32 index2 = scoresBegin != scoresEnd ? indexIterator[*scoresBegin] : numLabels;

                    if (index1 < index2) {
                        row.emplace_back(index1);
                        i++;
                    } else if (index1 == index2) {
                        row.emplace_back(index1);
                        i++;
                        scoresBegin++;
                    } else {
                        row.emplace_back(index2);
                        scoresBegin++;
                    }
                }

                delete[] buffer;
            }

            for (; scoresBegin != scoresEnd; scoresBegin++) {
                row.emplace_back(indexIterator[*scoresBegin]);
            }

            return row.size() - numElements;
        }

        return 0;
    }

    static inline uint32 applyHead(const IHead& head, BinaryLilMatrix::Row& row, uint32 numLabels) {
        uint32 numNonZeroElements;
        auto completeHeadVisitor = [&](const CompleteHead& head) mutable {
            numNonZeroElements = applyHead(
                make_non_zero_index_forward_iterator(head.scores_cbegin(), head.scores_cend()),
                make_non_zero_index_forward_iterator(head.scores_cend(), head.scores_cend()), IndexIterator(0), row,
                numLabels);
        };
        auto partialHeadVisitor = [&](const PartialHead& head) mutable {
            numNonZeroElements = applyHead(
                make_non_zero_index_forward_iterator(head.scores_cbegin(), head.scores_cend()),
                make_non_zero_index_forward_iterator(head.scores_cend(), head.scores_cend()), head.indices_cbegin(),
                row, numLabels);
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

            std::unique_ptr<DensePredictionMatrix<uint8>> predict(
                    const CContiguousConstView<const float32>& featureMatrix, uint32 numLabels) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                std::unique_ptr<DensePredictionMatrix<uint8>> predictionMatrixPtr =
                    std::make_unique<DensePredictionMatrix<uint8>>(numExamples, numLabels,
                                                                   !model_.containsDefaultRule());
                const CContiguousConstView<const float32>* featureMatrixPtr = &featureMatrix;
                CContiguousView<uint8>* predictionMatrixRawPtr = predictionMatrixPtr.get();
                const Model* modelPtr = &model_;

                #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(modelPtr) \
                firstprivate(featureMatrixPtr) firstprivate(predictionMatrixRawPtr) schedule(dynamic) \
                num_threads(numThreads_)
                for (int64 i = 0; i < numExamples; i++) {
                    BitVector mask(numLabels, true);

                    for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                        const RuleList::Rule& rule = *it;
                        const IBody& body = rule.getBody();

                        if (body.covers(featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i))) {
                            const IHead& head = rule.getHead();
                            applyHead(head, *predictionMatrixRawPtr, mask, i);
                        }
                    }
                }

                return predictionMatrixPtr;
            }

            std::unique_ptr<DensePredictionMatrix<uint8>> predict(const CsrConstView<const float32>& featureMatrix,
                                                                  uint32 numLabels) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                uint32 numFeatures = featureMatrix.getNumCols();
                std::unique_ptr<DensePredictionMatrix<uint8>> predictionMatrixPtr =
                    std::make_unique<DensePredictionMatrix<uint8>>(numExamples, numLabels,
                                                                   !model_.containsDefaultRule());
                const CsrConstView<const float32>* featureMatrixPtr = &featureMatrix;
                CContiguousView<uint8>* predictionMatrixRawPtr = predictionMatrixPtr.get();
                const Model* modelPtr = &model_;

                #pragma omp parallel for firstprivate(numExamples) firstprivate(numFeatures) firstprivate(numLabels) \
                firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixRawPtr) \
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
                            applyHead(head, *predictionMatrixRawPtr, mask, i);
                        }

                        n++;
                    }

                    delete[] tmpArray1;
                    delete[] tmpArray2;
                }

                return predictionMatrixPtr;
            }

            std::unique_ptr<BinarySparsePredictionMatrix> predictSparse(
                    const CContiguousConstView<const float32>& featureMatrix, uint32 numLabels) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                BinaryLilMatrix lilMatrix(numExamples);
                const CContiguousConstView<const float32>* featureMatrixPtr = &featureMatrix;
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

                        if (body.covers(featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i))) {
                            const IHead& head = rule.getHead();
                            numNonZeroElements += applyHead(head, row, numLabels);
                        }
                    }
                }

                return createBinarySparsePredictionMatrix(lilMatrix, numLabels, numNonZeroElements);
            }

            std::unique_ptr<BinarySparsePredictionMatrix> predictSparse(
                    const CsrConstView<const float32>& featureMatrix, uint32 numLabels) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                uint32 numFeatures = featureMatrix.getNumCols();
                BinaryLilMatrix lilMatrix(numExamples);
                const CsrConstView<const float32>* featureMatrixPtr = &featureMatrix;
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
                            numNonZeroElements += applyHead(head, row, numLabels);
                        }

                        n++;
                    }

                    delete[] tmpArray1;
                    delete[] tmpArray2;
                }

                return createBinarySparsePredictionMatrix(lilMatrix, numLabels, numNonZeroElements);
            }

    };

    /**
     * Allows to create instances of the class `IClassificationPredictor` that allow to predict whether individual
     * labels of given query examples are relevant or irrelevant by processing rules of an existing rule-based model in
     * the order they have been learned. If a rule covers an example, its prediction (1 if the label is relevant, 0
     * otherwise) is applied to each label individually, if none of the previous rules has already predicted for a
     * particular example and label.
     */
    class LabelWiseClassificationPredictorFactory final : public IClassificationPredictorFactory {

        private:

            uint32 numThreads_;

        public:

            /**
             * @param numThreads The number of CPU threads to be used to make predictions for different query examples
             *                   in parallel. Must be at least 1
             */
            LabelWiseClassificationPredictorFactory(uint32 numThreads)
                : numThreads_(numThreads) {

            }

            std::unique_ptr<IClassificationPredictor> create(const RuleList& model,
                                                             const LabelVectorSet* labelVectorSet) const override {
                return std::make_unique<LabelWiseClassificationPredictor<RuleList>>(model, numThreads_);
            }

    };

    LabelWiseClassificationPredictorConfig::LabelWiseClassificationPredictorConfig(
            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : multiThreadingConfigPtr_(multiThreadingConfigPtr) {

    }

    std::unique_ptr<IClassificationPredictorFactory> LabelWiseClassificationPredictorConfig::createClassificationPredictorFactory(
            const IFeatureMatrix& featureMatrix, const uint32 numLabels) const {
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
        return std::make_unique<LabelWiseClassificationPredictorFactory>(numThreads);
    }

    bool LabelWiseClassificationPredictorConfig::isLabelVectorSetNeeded() const {
        return false;
    }

}
