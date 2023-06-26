#include "seco/prediction/predictor_binary_label_wise.hpp"

#include "common/data/vector_bit.hpp"
#include "common/iterator/index_iterator.hpp"
#include "common/iterator/non_zero_index_forward_iterator.hpp"
#include "common/model/head_complete.hpp"
#include "common/model/head_partial.hpp"
#include "common/prediction/predictor_common.hpp"

#include <stdexcept>

namespace seco {

    static inline void applyHead(const CompleteHead& head, VectorView<uint8>::iterator iterator, BitVector& mask) {
        CompleteHead::score_const_iterator scoreIterator = head.scores_cbegin();
        uint32 numElements = head.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            if (!mask[i]) {
                uint8 prediction = scoreIterator[i] > 0;
                iterator[i] = prediction;
                mask.set(i, true);
            }
        }
    }

    static inline void applyHead(const PartialHead& head, VectorView<uint8>::iterator iterator, BitVector& mask) {
        PartialHead::score_const_iterator scoreIterator = head.scores_cbegin();
        PartialHead::index_const_iterator indexIterator = head.indices_cbegin();
        uint32 numElements = head.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];

            if (!mask[index]) {
                uint8 prediction = scoreIterator[i] > 0;
                iterator[index] = prediction;
                mask.set(index, true);
            }
        }
    }

    static inline void applyHead(const IHead& head, VectorView<uint8>::iterator scoreIterator, BitVector& mask) {
        auto completeHeadVisitor = [&](const CompleteHead& head) {
            applyHead(head, scoreIterator, mask);
        };
        auto partialHeadVisitor = [&](const PartialHead& head) {
            applyHead(head, scoreIterator, mask);
        };
        head.visit(completeHeadVisitor, partialHeadVisitor);
    }

    static inline void predictForExampleInternally(const CContiguousConstView<const float32>& featureMatrix,
                                                   RuleList::const_iterator rulesBegin,
                                                   RuleList::const_iterator rulesEnd,
                                                   CContiguousView<uint8>& predictionMatrix, uint32 exampleIndex,
                                                   uint32 predictionIndex) {
        uint32 numLabels = predictionMatrix.getNumCols();
        BitVector mask(numLabels, true);

        for (; rulesBegin != rulesEnd; rulesBegin++) {
            const RuleList::Rule& rule = *rulesBegin;
            const IBody& body = rule.getBody();

            if (body.covers(featureMatrix.values_cbegin(exampleIndex), featureMatrix.values_cend(exampleIndex))) {
                const IHead& head = rule.getHead();
                applyHead(head, predictionMatrix.values_begin(predictionIndex), mask);
            }
        }
    }

    static inline void predictForExampleInternally(const CsrConstView<const float32>& featureMatrix,
                                                   RuleList::const_iterator rulesBegin,
                                                   RuleList::const_iterator rulesEnd,
                                                   CContiguousView<uint8>& predictionMatrix, uint32 exampleIndex,
                                                   uint32 predictionIndex) {
        uint32 numFeatures = featureMatrix.getNumCols();
        uint32 numLabels = predictionMatrix.getNumCols();
        BitVector mask(numLabels, true);
        float32* tmpArray1 = new float32[numFeatures];
        uint32* tmpArray2 = new uint32[numFeatures] {};
        uint32 n = 1;

        for (; rulesBegin != rulesEnd; rulesBegin++) {
            const RuleList::Rule& rule = *rulesBegin;
            const IBody& body = rule.getBody();

            if (body.covers(featureMatrix.indices_cbegin(exampleIndex), featureMatrix.indices_cend(exampleIndex),
                            featureMatrix.values_cbegin(exampleIndex), featureMatrix.values_cend(exampleIndex),
                            &tmpArray1[0], &tmpArray2[0], n)) {
                const IHead& head = rule.getHead();
                applyHead(head, predictionMatrix.values_begin(predictionIndex), mask);
            }

            n++;
        }

        delete[] tmpArray1;
        delete[] tmpArray2;
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

            class PredictionDelegate final
                : public PredictionDispatcher<uint8, FeatureMatrix, Model>::IPredictionDelegate {
                private:

                    CContiguousView<uint8>& predictionMatrix_;

                public:

                    PredictionDelegate(CContiguousView<uint8>& predictionMatrix)
                        : predictionMatrix_(predictionMatrix) {}

                    void predictForExample(const FeatureMatrix& featureMatrix,
                                           typename Model::const_iterator rulesBegin,
                                           typename Model::const_iterator rulesEnd, uint32 threadIndex,
                                           uint32 exampleIndex, uint32 predictionIndex) const override {
                        predictForExampleInternally(featureMatrix, rulesBegin, rulesEnd, predictionMatrix_,
                                                    exampleIndex, predictionIndex);
                    }
            };

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            const uint32 numLabels_;

            const uint32 numThreads_;

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

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<uint8>> predict(uint32 maxRules) const override {
                uint32 numExamples = featureMatrix_.getNumRows();
                std::unique_ptr<DensePredictionMatrix<uint8>> predictionMatrixPtr =
                  std::make_unique<DensePredictionMatrix<uint8>>(numExamples, numLabels_,
                                                                 !model_.containsDefaultRule());
                PredictionDelegate delegate(*predictionMatrixPtr);
                PredictionDispatcher<uint8, FeatureMatrix, Model>().predict(
                  delegate, featureMatrix_, model_.used_cbegin(maxRules), model_.used_cend(maxRules), numThreads_);
                return predictionMatrixPtr;
            }

            /**
             * @see `IPredictor::canPredictIncrementally`
             */
            bool canPredictIncrementally() const override {
                return false;
            }

            /**
             * @see `IPredictor::createIncrementalPredictor`
             */
            std::unique_ptr<IIncrementalPredictor<DensePredictionMatrix<uint8>>> createIncrementalPredictor(
              uint32 maxRules) const override {
                throw std::runtime_error("The rule learner does not support to predict binary labels incrementally");
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

            const uint32 numThreads_;

        public:

            /**
             * @param numThreads The number of CPU threads to be used to make predictions for different query examples
             *                   in parallel. Must be at least 1
             */
            LabelWiseBinaryPredictorFactory(uint32 numThreads) : numThreads_(numThreads) {}

            std::unique_ptr<IBinaryPredictor> create(
              const CContiguousConstView<const float32>& featureMatrix, const RuleList& model,
              const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
              uint32 numLabels) const override {
                return std::make_unique<LabelWiseBinaryPredictor<CContiguousConstView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, numThreads_);
            }

            std::unique_ptr<IBinaryPredictor> create(
              const CsrConstView<const float32>& featureMatrix, const RuleList& model,
              const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
              uint32 numLabels) const override {
                return std::make_unique<LabelWiseBinaryPredictor<CsrConstView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, numThreads_);
            }
    };

    template<typename ScoreIterator, typename IndexIterator>
    static inline void applyHead(ScoreIterator scoresBegin, ScoreIterator scoresEnd, IndexIterator indexIterator,
                                 BinaryLilMatrix::row predictionRow, uint32 numLabels) {
        if (scoresBegin != scoresEnd) {
            if (predictionRow.size() > 0) {
                BinaryLilMatrix::iterator end = predictionRow.end();
                BinaryLilMatrix::iterator start =
                  std::lower_bound(predictionRow.begin(), end, indexIterator[*scoresBegin]);
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
                        predictionRow.emplace_back(index1);
                        i++;
                    } else if (index1 == index2) {
                        predictionRow.emplace_back(index1);
                        i++;
                        scoresBegin++;
                    } else {
                        predictionRow.emplace_back(index2);
                        scoresBegin++;
                    }
                }

                delete[] buffer;
            }

            for (; scoresBegin != scoresEnd; scoresBegin++) {
                predictionRow.emplace_back(indexIterator[*scoresBegin]);
            }
        }
    }

    static inline void applyHead(const IHead& head, BinaryLilMatrix::row predictionRow, uint32 numLabels) {
        auto completeHeadVisitor = [&](const CompleteHead& head) {
            applyHead(make_non_zero_index_forward_iterator(head.scores_cbegin(), head.scores_cend()),
                      make_non_zero_index_forward_iterator(head.scores_cend(), head.scores_cend()), IndexIterator(0),
                      predictionRow, numLabels);
        };
        auto partialHeadVisitor = [&](const PartialHead& head) {
            applyHead(make_non_zero_index_forward_iterator(head.scores_cbegin(), head.scores_cend()),
                      make_non_zero_index_forward_iterator(head.scores_cend(), head.scores_cend()),
                      head.indices_cbegin(), predictionRow, numLabels);
        };
        head.visit(completeHeadVisitor, partialHeadVisitor);
    }

    static inline void predictForExampleInternally(const CContiguousConstView<const float32>& featureMatrix,
                                                   RuleList::const_iterator rulesBegin,
                                                   RuleList::const_iterator rulesEnd,
                                                   BinaryLilMatrix::row predictionRow, uint32 numLabels,
                                                   uint32 exampleIndex) {
        for (; rulesBegin != rulesEnd; rulesBegin++) {
            const RuleList::Rule& rule = *rulesBegin;
            const IBody& body = rule.getBody();

            if (body.covers(featureMatrix.values_cbegin(exampleIndex), featureMatrix.values_cend(exampleIndex))) {
                const IHead& head = rule.getHead();
                applyHead(head, predictionRow, numLabels);
            }
        }
    }

    static inline void predictForExampleInternally(const CsrConstView<const float32>& featureMatrix,
                                                   RuleList::const_iterator rulesBegin,
                                                   RuleList::const_iterator rulesEnd,
                                                   BinaryLilMatrix::row predictionRow, uint32 numLabels,
                                                   uint32 exampleIndex) {
        uint32 numFeatures = featureMatrix.getNumCols();
        float32* tmpArray1 = new float32[numFeatures];
        uint32* tmpArray2 = new uint32[numFeatures] {};
        uint32 n = 1;

        for (; rulesBegin != rulesEnd; rulesBegin++) {
            const RuleList::Rule& rule = *rulesBegin;
            const IBody& body = rule.getBody();

            if (body.covers(featureMatrix.indices_cbegin(exampleIndex), featureMatrix.indices_cend(exampleIndex),
                            featureMatrix.values_cbegin(exampleIndex), featureMatrix.values_cend(exampleIndex),
                            &tmpArray1[0], &tmpArray2[0], n)) {
                const IHead& head = rule.getHead();
                applyHead(head, predictionRow, numLabels);
            }

            n++;
        }

        delete[] tmpArray1;
        delete[] tmpArray2;
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

            typedef BinarySparsePredictionDispatcher<FeatureMatrix, Model> Dispatcher;

            class Delegate final : public Dispatcher::IPredictionDelegate {
                private:

                    BinaryLilMatrix& predictionMatrix_;

                    const uint32 numLabels_;

                public:

                    Delegate(BinaryLilMatrix& predictionMatrix, uint32 numLabels)
                        : predictionMatrix_(predictionMatrix), numLabels_(numLabels) {}

                    uint32 predictForExample(const FeatureMatrix& featureMatrix,
                                             typename Model::const_iterator rulesBegin,
                                             typename Model::const_iterator rulesEnd, uint32 threadIndex,
                                             uint32 exampleIndex, uint32 predictionIndex) const override {
                        BinaryLilMatrix::row predictionRow = predictionMatrix_[predictionIndex];
                        predictForExampleInternally(featureMatrix, rulesBegin, rulesEnd, predictionRow, numLabels_,
                                                    exampleIndex);
                        return (uint32) predictionRow.size();
                    }
            };

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            const uint32 numLabels_;

            const uint32 numThreads_;

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

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<BinarySparsePredictionMatrix> predict(uint32 maxRules) const override {
                uint32 numExamples = featureMatrix_.getNumRows();
                BinaryLilMatrix predictionMatrix(numExamples);
                Delegate delegate(predictionMatrix, numLabels_);
                uint32 numNonZeroElements = Dispatcher().predict(delegate, featureMatrix_, model_.used_cbegin(maxRules),
                                                                 model_.used_cend(maxRules), numThreads_);
                return createBinarySparsePredictionMatrix(predictionMatrix, numLabels_, numNonZeroElements);
            }

            /**
             * @see `IPredictor::canPredictIncrementally`
             */
            bool canPredictIncrementally() const override {
                return false;
            }

            /**
             * @see `IPredictor::createIncrementalPredictor`
             */
            std::unique_ptr<IIncrementalPredictor<BinarySparsePredictionMatrix>> createIncrementalPredictor(
              uint32 maxRules) const override {
                throw std::runtime_error(
                  "The rule learner does not support to predict sparse binary labels incrementally");
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

            const uint32 numThreads_;

        public:

            /**
             * @param numThreads The number of CPU threads to be used to make predictions for different query examples
             *                   in parallel. Must be at least 1
             */
            LabelWiseSparseBinaryPredictorFactory(uint32 numThreads) : numThreads_(numThreads) {}

            std::unique_ptr<ISparseBinaryPredictor> create(
              const CContiguousConstView<const float32>& featureMatrix, const RuleList& model,
              const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
              uint32 numLabels) const override {
                return std::make_unique<LabelWiseSparseBinaryPredictor<CContiguousConstView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, numThreads_);
            }

            std::unique_ptr<ISparseBinaryPredictor> create(
              const CsrConstView<const float32>& featureMatrix, const RuleList& model,
              const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
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
