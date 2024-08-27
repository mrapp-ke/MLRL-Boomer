#include "mlrl/seco/prediction/predictor_binary_output_wise.hpp"

#include "mlrl/common/data/array.hpp"
#include "mlrl/common/data/vector_bit.hpp"
#include "mlrl/common/iterator/iterator_forward_non_zero_index.hpp"
#include "mlrl/common/iterator/iterator_index.hpp"
#include "mlrl/common/model/head_complete.hpp"
#include "mlrl/common/model/head_partial.hpp"
#include "mlrl/common/prediction/predictor_common.hpp"

namespace seco {

    static inline void applyHead(const CompleteHead& head, View<uint8>::iterator iterator, BitVector& mask) {
        CompleteHead::value_const_iterator valueIterator = head.values_cbegin();
        uint32 numElements = head.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            if (!mask[i]) {
                uint8 prediction = valueIterator[i] > 0;
                iterator[i] = prediction;
                mask.set(i, true);
            }
        }
    }

    static inline void applyHead(const PartialHead& head, View<uint8>::iterator iterator, BitVector& mask) {
        PartialHead::value_const_iterator valueIterator = head.values_cbegin();
        PartialHead::index_const_iterator indexIterator = head.indices_cbegin();
        uint32 numElements = head.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];

            if (!mask[index]) {
                uint8 prediction = valueIterator[i] > 0;
                iterator[index] = prediction;
                mask.set(index, true);
            }
        }
    }

    static inline void applyHead(const IHead& head, View<uint8>::iterator scoreIterator, BitVector& mask) {
        auto completeHeadVisitor = [&](const CompleteHead& head) {
            applyHead(head, scoreIterator, mask);
        };
        auto partialHeadVisitor = [&](const PartialHead& head) {
            applyHead(head, scoreIterator, mask);
        };
        head.visit(completeHeadVisitor, partialHeadVisitor);
    }

    static inline void predictForExampleInternally(const CContiguousView<const float32>& featureMatrix,
                                                   RuleList::const_iterator rulesBegin,
                                                   RuleList::const_iterator rulesEnd,
                                                   CContiguousView<uint8>& predictionMatrix, uint32 exampleIndex,
                                                   uint32 predictionIndex) {
        BitVector mask(predictionMatrix.numCols, true);

        for (; rulesBegin != rulesEnd; rulesBegin++) {
            const RuleList::Rule& rule = *rulesBegin;
            const IBody& body = rule.getBody();

            if (body.covers(featureMatrix.values_cbegin(exampleIndex), featureMatrix.values_cend(exampleIndex))) {
                const IHead& head = rule.getHead();
                applyHead(head, predictionMatrix.values_begin(predictionIndex), mask);
            }
        }
    }

    static inline void predictForExampleInternally(const CsrView<const float32>& featureMatrix,
                                                   RuleList::const_iterator rulesBegin,
                                                   RuleList::const_iterator rulesEnd,
                                                   CContiguousView<uint8>& predictionMatrix, uint32 exampleIndex,
                                                   uint32 predictionIndex) {
        BitVector mask(predictionMatrix.numCols, true);
        uint32 numFeatures = featureMatrix.numCols;
        Array<float32> tmpArray1(numFeatures);
        Array<uint32> tmpArray2(numFeatures, true);
        uint32 n = 1;

        for (; rulesBegin != rulesEnd; rulesBegin++) {
            const RuleList::Rule& rule = *rulesBegin;
            const IBody& body = rule.getBody();

            if (body.covers(featureMatrix.indices_cbegin(exampleIndex), featureMatrix.indices_cend(exampleIndex),
                            featureMatrix.values_cbegin(exampleIndex), featureMatrix.values_cend(exampleIndex),
                            featureMatrix.sparseValue, tmpArray1.begin(), tmpArray2.begin(), n)) {
                const IHead& head = rule.getHead();
                applyHead(head, predictionMatrix.values_begin(predictionIndex), mask);
            }

            n++;
        }
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
    class OutputWiseBinaryPredictor final : public IBinaryPredictor {
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

            const MultiThreadingSettings multiThreadingSettings_;

        public:

            /**
             * @param featureMatrix             A reference to an object of template type `FeatureMatrix` that provides
             *                                  row-wise access to the feature values of the query examples
             * @param model                     A reference to an object of template type `Model` that should be used to
             *                                  obtain predictions
             * @param multiThreadingSettings    An object of type `MultiThreadingSettings` that stores the settings to
             *                                  be used for making predictions for different query examples in parallel
             */
            OutputWiseBinaryPredictor(const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels,
                                      MultiThreadingSettings multiThreadingSettings)
                : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels),
                  multiThreadingSettings_(multiThreadingSettings) {}

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<uint8>> predict(uint32 maxRules) const override {
                std::unique_ptr<DensePredictionMatrix<uint8>> predictionMatrixPtr =
                  std::make_unique<DensePredictionMatrix<uint8>>(featureMatrix_.numRows, numLabels_,
                                                                 !model_.containsDefaultRule());
                PredictionDelegate delegate(predictionMatrixPtr->getView());
                PredictionDispatcher<uint8, FeatureMatrix, Model>().predict(
                  delegate, featureMatrix_, model_.used_cbegin(maxRules), model_.used_cend(maxRules),
                  multiThreadingSettings_);
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
    class OutputWiseBinaryPredictorFactory final : public IBinaryPredictorFactory {
        private:

            const MultiThreadingSettings multiThreadingSettings_;

        public:

            /**
             * @param multiThreadingSettings An object of type `MultiThreadingSettings` that stores the settings to be
             *                               used for making predictions for different query examples in parallel
             */
            OutputWiseBinaryPredictorFactory(MultiThreadingSettings multiThreadingSettings)
                : multiThreadingSettings_(multiThreadingSettings) {}

            std::unique_ptr<IBinaryPredictor> create(
              const CContiguousView<const float32>& featureMatrix, const RuleList& model,
              const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
              uint32 numLabels) const override {
                return std::make_unique<OutputWiseBinaryPredictor<CContiguousView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, multiThreadingSettings_);
            }

            std::unique_ptr<IBinaryPredictor> create(
              const CsrView<const float32>& featureMatrix, const RuleList& model, const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
              uint32 numLabels) const override {
                return std::make_unique<OutputWiseBinaryPredictor<CsrView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, multiThreadingSettings_);
            }
    };

    template<typename ScoreIterator, typename IndexIterator>
    static inline void applyHead(ScoreIterator scoresBegin, ScoreIterator scoresEnd, IndexIterator indexIterator,
                                 BinaryLilMatrix::row predictionRow, uint32 numLabels) {
        if (scoresBegin != scoresEnd) {
            if (predictionRow.size() > 0) {
                BinaryLilMatrix::value_iterator end = predictionRow.end();
                BinaryLilMatrix::value_iterator start =
                  std::lower_bound(predictionRow.begin(), end, indexIterator[*scoresBegin]);
                uint32 bufferSize = end - start;
                Array<uint32> buffer(bufferSize);
                util::copyView(start, buffer.begin(), bufferSize);
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
            }

            for (; scoresBegin != scoresEnd; scoresBegin++) {
                predictionRow.emplace_back(indexIterator[*scoresBegin]);
            }
        }
    }

    static inline void applyHead(const IHead& head, BinaryLilMatrix::row predictionRow, uint32 numLabels) {
        auto completeHeadVisitor = [&](const CompleteHead& head) {
            applyHead(createNonZeroIndexForwardIterator(head.values_cbegin(), head.values_cend()),
                      createNonZeroIndexForwardIterator(head.values_cend(), head.values_cend()), IndexIterator(0),
                      predictionRow, numLabels);
        };
        auto partialHeadVisitor = [&](const PartialHead& head) {
            applyHead(createNonZeroIndexForwardIterator(head.values_cbegin(), head.values_cend()),
                      createNonZeroIndexForwardIterator(head.values_cend(), head.values_cend()), head.indices_cbegin(),
                      predictionRow, numLabels);
        };
        head.visit(completeHeadVisitor, partialHeadVisitor);
    }

    static inline void predictForExampleInternally(const CContiguousView<const float32>& featureMatrix,
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

    static inline void predictForExampleInternally(const CsrView<const float32>& featureMatrix,
                                                   RuleList::const_iterator rulesBegin,
                                                   RuleList::const_iterator rulesEnd,
                                                   BinaryLilMatrix::row predictionRow, uint32 numLabels,
                                                   uint32 exampleIndex) {
        uint32 numFeatures = featureMatrix.numCols;
        Array<float32> tmpArray1(numFeatures);
        Array<uint32> tmpArray2(numFeatures, true);
        uint32 n = 1;

        for (; rulesBegin != rulesEnd; rulesBegin++) {
            const RuleList::Rule& rule = *rulesBegin;
            const IBody& body = rule.getBody();

            if (body.covers(featureMatrix.indices_cbegin(exampleIndex), featureMatrix.indices_cend(exampleIndex),
                            featureMatrix.values_cbegin(exampleIndex), featureMatrix.values_cend(exampleIndex),
                            featureMatrix.sparseValue, tmpArray1.begin(), tmpArray2.begin(), n)) {
                const IHead& head = rule.getHead();
                applyHead(head, predictionRow, numLabels);
            }

            n++;
        }
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
    class OutputWiseSparseBinaryPredictor final : public ISparseBinaryPredictor {
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
                        return static_cast<uint32>(predictionRow.size());
                    }
            };

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            const uint32 numLabels_;

            const MultiThreadingSettings multiThreadingSettings_;

        public:

            /**
             * @param featureMatrix             A reference to an object of template type `FeatureMatrix` that provides
             *                                  row-wise access to the feature values of the query examples
             * @param model                     A reference to an object of template type `Model` that should be used to
             *                                  obtain predictions
             * @param multiThreadnigSettings    An object of type `MultiThreadingSettings` that stores the settings to
             *                                  be used for making predictions for different query examples in parallel
             */
            OutputWiseSparseBinaryPredictor(const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels,
                                            MultiThreadingSettings multiThreadingSettings)
                : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels),
                  multiThreadingSettings_(multiThreadingSettings) {}

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<BinarySparsePredictionMatrix> predict(uint32 maxRules) const override {
                BinaryLilMatrix predictionMatrix(featureMatrix_.numRows, numLabels_);
                Delegate delegate(predictionMatrix, numLabels_);
                uint32 numDenseElements = Dispatcher().predict(delegate, featureMatrix_, model_.used_cbegin(maxRules),
                                                               model_.used_cend(maxRules), multiThreadingSettings_);
                return createBinarySparsePredictionMatrix(predictionMatrix, numLabels_, numDenseElements);
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
    class OutputWiseSparseBinaryPredictorFactory final : public ISparseBinaryPredictorFactory {
        private:

            const MultiThreadingSettings multiThreadingSettings_;

        public:

            /**
             * @param multiThreadingSettings An object of type `MultiThreadingSettings` that stores the settings to be
             *                               used for making predictions for different query examples in parallel
             */
            OutputWiseSparseBinaryPredictorFactory(MultiThreadingSettings multiThreadingSettings)
                : multiThreadingSettings_(multiThreadingSettings) {}

            std::unique_ptr<ISparseBinaryPredictor> create(
              const CContiguousView<const float32>& featureMatrix, const RuleList& model,
              const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
              uint32 numLabels) const override {
                return std::make_unique<OutputWiseSparseBinaryPredictor<CContiguousView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, multiThreadingSettings_);
            }

            std::unique_ptr<ISparseBinaryPredictor> create(
              const CsrView<const float32>& featureMatrix, const RuleList& model, const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
              uint32 numLabels) const override {
                return std::make_unique<OutputWiseSparseBinaryPredictor<CsrView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, multiThreadingSettings_);
            }
    };

    OutputWiseBinaryPredictorConfig::OutputWiseBinaryPredictorConfig(
      ReadableProperty<IMultiThreadingConfig> multiThreadingConfig)
        : multiThreadingConfig_(multiThreadingConfig) {}

    std::unique_ptr<IBinaryPredictorFactory> OutputWiseBinaryPredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, const uint32 numOutputs) const {
        MultiThreadingSettings multiThreadingSettings =
          multiThreadingConfig_.get().getSettings(featureMatrix, numOutputs);
        return std::make_unique<OutputWiseBinaryPredictorFactory>(multiThreadingSettings);
    }

    std::unique_ptr<ISparseBinaryPredictorFactory> OutputWiseBinaryPredictorConfig::createSparsePredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, const uint32 numLabels) const {
        MultiThreadingSettings multiThreadingSettings =
          multiThreadingConfig_.get().getSettings(featureMatrix, numLabels);
        return std::make_unique<OutputWiseSparseBinaryPredictorFactory>(multiThreadingSettings);
    }

    bool OutputWiseBinaryPredictorConfig::isLabelVectorSetNeeded() const {
        return false;
    }

}
