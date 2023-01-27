#include "seco/prediction/predictor_binary_label_wise.hpp"

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
                                   BinaryLilMatrix::row row, uint32 numLabels) {
        if (scoresBegin != scoresEnd) {
            uint32 numElements = row.size();

            if (numElements > 0) {
                BinaryLilMatrix::iterator end = row.end();
                BinaryLilMatrix::iterator start = std::lower_bound(row.begin(), end, indexIterator[*scoresBegin]);
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

    static inline uint32 applyHead(const IHead& head, BinaryLilMatrix::row row, uint32 numLabels) {
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

    static inline std::unique_ptr<DensePredictionMatrix<uint8>> predictInternally(
            const CContiguousConstView<const float32>& featureMatrix, const RuleList& model, uint32 numLabels,
            uint32 numThreads) {
        uint32 numExamples = featureMatrix.getNumRows();
        std::unique_ptr<DensePredictionMatrix<uint8>> predictionMatrixPtr =
            std::make_unique<DensePredictionMatrix<uint8>>(numExamples, numLabels, !model.containsDefaultRule());
        const CContiguousConstView<const float32>* featureMatrixPtr = &featureMatrix;
        CContiguousView<uint8>* predictionMatrixRawPtr = predictionMatrixPtr.get();
        const RuleList* modelPtr = &model;

        #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(modelPtr) \
        firstprivate(featureMatrixPtr) firstprivate(predictionMatrixRawPtr) schedule(dynamic) num_threads(numThreads)
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

    static inline std::unique_ptr<DensePredictionMatrix<uint8>> predictInternally(
            const CsrConstView<const float32>& featureMatrix, const RuleList& model, uint32 numLabels,
            uint32 numThreads) {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numFeatures = featureMatrix.getNumCols();
        std::unique_ptr<DensePredictionMatrix<uint8>> predictionMatrixPtr =
            std::make_unique<DensePredictionMatrix<uint8>>(numExamples, numLabels, !model.containsDefaultRule());
        const CsrConstView<const float32>* featureMatrixPtr = &featureMatrix;
        CContiguousView<uint8>* predictionMatrixRawPtr = predictionMatrixPtr.get();
        const RuleList* modelPtr = &model;

        #pragma omp parallel for firstprivate(numExamples) firstprivate(numFeatures) firstprivate(numLabels) \
        firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixRawPtr) schedule(dynamic) \
        num_threads(numThreads)
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

    /**
     * An implementation of the type `IBinaryPredictor` that allows to predict whether individual labels of given query
     * examples are relevant or irrelevant by processing rules of an existing rule-based model in the order they have
     * been learned. If a rule covers an example, its prediction (1 if the label is relevant, 0 otherwise) is applied to
     * each label individually, if none of the previous rules has already predicted for a particular example and label.
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class LabelWiseBinaryPredictor final : public IBinaryPredictor {
        private:

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            uint32 numLabels_;

            uint32 numThreads_;

        public:

            /**
             * @param featureMatrix A reference to an object of template type `FeatureMatrix` that provides row-wise
             *                      access to the feature values of the query examples
             * @param model         A reference to an object of template type `Model` that should be used to obtain
             *                      predictions
             * @param numThreads    The number of CPU threads to be used to make predictions for different query
             *                      examples in parallel. Must be at least 1
             */
            LabelWiseBinaryPredictor(const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels,
                                     uint32 numThreads)
                : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels), numThreads_(numThreads) {}

            std::unique_ptr<DensePredictionMatrix<uint8>> predict() const override {
                return predictInternally(featureMatrix_, model_, numLabels_, numThreads_);
            }
    };

    /**
     * Allows to create instances of the class `IBinaryPredictor` that allow to predict whether individual labels of
     * given query examples are relevant or irrelevant by processing rules of an existing rule-based model in the order
     * they have been learned. If a rule covers an example, its prediction (1 if the label is relevant, 0 otherwise) is
     * applied to each label individually, if none of the previous rules has already predicted for a particular example
     * and label.
     */
    class LabelWiseBinaryPredictorFactory final : public IBinaryPredictorFactory {
        private:

            uint32 numThreads_;

        public:

            /**
             * @param numThreads The number of CPU threads to be used to make predictions for different query examples
             *                   in parallel. Must be at least 1
             */
            LabelWiseBinaryPredictorFactory(uint32 numThreads)
                : numThreads_(numThreads) {}

            std::unique_ptr<IBinaryPredictor> create(const CContiguousConstView<const float32>& featureMatrix,
                                                     const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                     uint32 numLabels) const override {
                return std::make_unique<LabelWiseBinaryPredictor<CContiguousConstView<const float32>, RuleList>>(
                    featureMatrix, model, numLabels, numThreads_);
            }

            std::unique_ptr<IBinaryPredictor> create(const CsrConstView<const float32>& featureMatrix,
                                                     const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                     uint32 numLabels) const override {
                return std::make_unique<LabelWiseBinaryPredictor<CsrConstView<const float32>, RuleList>>(
                    featureMatrix, model, numLabels, numThreads_);
            }
    };

    static inline std::unique_ptr<BinarySparsePredictionMatrix> predictSparseInternally(
            const CContiguousConstView<const float32>& featureMatrix, const RuleList& model, uint32 numLabels,
            uint32 numThreads) {
        uint32 numExamples = featureMatrix.getNumRows();
        BinaryLilMatrix lilMatrix(numExamples);
        const CContiguousConstView<const float32>* featureMatrixPtr = &featureMatrix;
        BinaryLilMatrix* predictionMatrixPtr = &lilMatrix;
        const RuleList* modelPtr = &model;
        uint32 numNonZeroElements = 0;

        #pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) firstprivate(numLabels) \
        firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) schedule(dynamic) \
        num_threads(numThreads)
        for (int64 i = 0; i < numExamples; i++) {
            BinaryLilMatrix::row row = (*predictionMatrixPtr)[i];

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

    static inline std::unique_ptr<BinarySparsePredictionMatrix> predictSparseInternally(
            const CsrConstView<const float32>& featureMatrix, const RuleList& model, uint32 numLabels,
            uint32 numThreads) {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numFeatures = featureMatrix.getNumCols();
        BinaryLilMatrix lilMatrix(numExamples);
        const CsrConstView<const float32>* featureMatrixPtr = &featureMatrix;
        BinaryLilMatrix* predictionMatrixPtr = &lilMatrix;
        const RuleList* modelPtr = &model;
        uint32 numNonZeroElements = 0;

        #pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) firstprivate(numFeatures) \
        firstprivate(numLabels) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
        firstprivate(predictionMatrixPtr) schedule(dynamic) num_threads(numThreads)
        for (int64 i = 0; i < numExamples; i++) {
            BinaryLilMatrix::row row = (*predictionMatrixPtr)[i];
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

    /**
     * An implementation of the type `ISparseBinaryPredictor` that allows to predict whether individual labels of given
     * query examples are relevant or irrelevant by processing rules of an existing rule-based model in the order they
     * have been learned. If a rule covers an example, its prediction (1 if the label is relevant, 0 otherwise) is
     * applied to each label individually, if none of the previous rules has already predicted for a particular example
     * and label.
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class LabelWiseSparseBinaryPredictor final : public ISparseBinaryPredictor {
        private:

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            uint32 numLabels_;

            uint32 numThreads_;

        public:

            /**
             * @param featureMatrix A reference to an object of template type `FeatureMatrix` that provides row-wise
             *                      access to the feature values of the query examples
             * @param model         A reference to an object of template type `Model` that should be used to obtain
             *                      predictions
             * @param numThreads    The number of CPU threads to be used to make predictions for different query
             *                      examples in parallel. Must be at least 1
             */
            LabelWiseSparseBinaryPredictor(const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels,
                                           uint32 numThreads)
                : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels), numThreads_(numThreads) {}

            std::unique_ptr<BinarySparsePredictionMatrix> predict() const override {
                return predictSparseInternally(featureMatrix_, model_, numLabels_, numThreads_);
            }
    };

    /**
     * Allows to create instances of the class `ISparseBinaryPredictor` that allow to predict whether individual labels
     * of given query examples are relevant or irrelevant by processing rules of an existing rule-based model in the
     * order they have been learned. If a rule covers an example, its prediction (1 if the label is relevant, 0
     * otherwise) is applied to each label individually, if none of the previous rules has already predicted for a
     * particular example and label.
     */
    class LabelWiseSparseBinaryPredictorFactory final : public ISparseBinaryPredictorFactory {
        private:

            uint32 numThreads_;

        public:

            /**
             * @param numThreads The number of CPU threads to be used to make predictions for different query examples
             *                   in parallel. Must be at least 1
             */
            LabelWiseSparseBinaryPredictorFactory(uint32 numThreads)
                : numThreads_(numThreads) {}

            std::unique_ptr<ISparseBinaryPredictor> create(const CContiguousConstView<const float32>& featureMatrix,
                                                           const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                           uint32 numLabels) const override {
                return std::make_unique<LabelWiseSparseBinaryPredictor<CContiguousConstView<const float32>, RuleList>>(
                    featureMatrix, model, numLabels, numThreads_);
            }

            std::unique_ptr<ISparseBinaryPredictor> create(const CsrConstView<const float32>& featureMatrix,
                                                           const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                           uint32 numLabels) const override {
                return std::make_unique<LabelWiseSparseBinaryPredictor<CsrConstView<const float32>, RuleList>>(
                    featureMatrix, model, numLabels, numThreads_);
            }
    };

    LabelWiseBinaryPredictorConfig::LabelWiseBinaryPredictorConfig(
            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : multiThreadingConfigPtr_(multiThreadingConfigPtr) {}

    std::unique_ptr<IBinaryPredictorFactory> LabelWiseBinaryPredictorConfig::createPredictorFactory(
            const IRowWiseFeatureMatrix& featureMatrix, const uint32 numLabels) const {
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
        return std::make_unique<LabelWiseBinaryPredictorFactory>(numThreads);
    }

    std::unique_ptr<ISparseBinaryPredictorFactory> LabelWiseBinaryPredictorConfig::createSparsePredictorFactory(
            const IRowWiseFeatureMatrix& featureMatrix, const uint32 numLabels) const {
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
        return std::make_unique<LabelWiseSparseBinaryPredictorFactory>(numThreads);
    }

    bool LabelWiseBinaryPredictorConfig::isLabelVectorSetNeeded() const {
        return false;
    }

}
