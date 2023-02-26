#include "boosting/prediction/predictor_binary_gfm.hpp"

#include "boosting/prediction/predictor_probability_common.hpp"
#include "common/data/arrays.hpp"
#include "common/data/matrix_sparse_set.hpp"
#include "common/data/vector_sparse_array.hpp"
#include "joint_probabilities.hpp"

#include <algorithm>
#include <stdexcept>

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
                                       DensePredictionMatrix<uint8>::value_iterator predictionIterator) {
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
    static inline void predictGfm(const float64* scoresBegin, Prediction prediction, uint32 numLabels,
                                  const IProbabilityFunction& probabilityFunction, const LabelVectorSet& labelVectorSet,
                                  uint32 numLabelVectors, uint32 maxLabelCardinality) {
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

    /**
     * An implementation of the type `IBinaryPredictor` that allows to predict whether individual labels of given query
     * examples are relevant or irrelevant by summing up the scores that are provided by the individual rules of an
     * existing rule-based model and transforming them into binary values according to the general F-measure maximizer
     * (GFM).
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class GfmBinaryPredictor final : public IBinaryPredictor {
        private:

            typedef PredictionDispatcher<uint8, FeatureMatrix, Model> Dispatcher;

            class Delegate final : public Dispatcher::IPredictionDelegate {
                private:

                    CContiguousView<float64>& scoreMatrix_;

                    CContiguousView<uint8>& predictionMatrix_;

                    const LabelVectorSet& labelVectorSet_;

                    const IProbabilityFunction& probabilityFunction_;

                    uint32 maxLabelCardinality_;

                public:

                    Delegate(CContiguousView<float64>& scoreMatrix, CContiguousView<uint8>& predictionMatrix,
                             const LabelVectorSet& labelVectorSet, const IProbabilityFunction& probabilityFunction,
                             uint32 maxLabelCardinality)
                        : scoreMatrix_(scoreMatrix), predictionMatrix_(predictionMatrix),
                          labelVectorSet_(labelVectorSet), probabilityFunction_(probabilityFunction),
                          maxLabelCardinality_(maxLabelCardinality) {}

                    void predictForExample(const FeatureMatrix& featureMatrix, const Model& model, uint32 maxRules,
                                           uint32 threadIndex, uint32 exampleIndex,
                                           uint32 predictionIndex) const override {
                        uint32 numLabels = scoreMatrix_.getNumCols();
                        CContiguousView<float64>::value_iterator scoreIterator =
                          scoreMatrix_.row_values_begin(threadIndex);
                        setArrayToZeros(scoreIterator, numLabels);
                        ScorePredictionDelegate<FeatureMatrix, Model>(scoreMatrix_)
                          .predictForExample(featureMatrix, model, maxRules, threadIndex, exampleIndex, threadIndex);
                        uint32 numLabelVectors = labelVectorSet_.getNumLabelVectors();
                        predictGfm(scoreIterator, predictionMatrix_.row_values_begin(predictionIndex), numLabels,
                                   probabilityFunction_, labelVectorSet_, numLabelVectors, maxLabelCardinality_);
                    }
            };

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            uint32 numLabels_;

            uint32 numThreads_;

            const LabelVectorSet& labelVectorSet_;

            std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr_;

        public:

            /**
             * @param featureMatrix             A reference to an object of template type `FeatureMatrix` that provide
             *                                  row-wise access to the feature values of the query examples
             * @param model                     A reference to an object of template type `Model` that should be used to
             *                                  obtain predictions
             * @param labelVectorSet            A reference to an object of type `LabelVectorSet` that stores all known
             *                                  label vectors
             * @param numLabels                 The number of labels to predict for
             * @param probabilityFunctionPtr    An unique pointer to an object of type `IProbabilityFunction` that
             *                                  should be used to transform predicted scores into probabilities
             * @param numThreads                The number of CPU threads to be used to make predictions for different
             *                                  query examples in parallel. Must be at least 1
             */
            GfmBinaryPredictor(const FeatureMatrix& featureMatrix, const Model& model,
                               const LabelVectorSet& labelVectorSet, uint32 numLabels,
                               std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr, uint32 numThreads)
                : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels), numThreads_(numThreads),
                  labelVectorSet_(labelVectorSet), probabilityFunctionPtr_(std::move(probabilityFunctionPtr)) {}

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<uint8>> predict(uint32 maxRules) const override {
                uint32 numExamples = featureMatrix_.getNumRows();
                std::unique_ptr<DensePredictionMatrix<uint8>> predictionMatrixPtr =
                  std::make_unique<DensePredictionMatrix<uint8>>(numExamples, numLabels_, true);

                if (labelVectorSet_.getNumLabelVectors() > 0) {
                    DenseMatrix<float64> scoreMatrix(numExamples, numLabels_);
                    uint32 maxLabelCardinality = getMaxLabelCardinality(labelVectorSet_);
                    Delegate delegate(scoreMatrix, *predictionMatrixPtr, labelVectorSet_, *probabilityFunctionPtr_,
                                      maxLabelCardinality);
                    Dispatcher().predict(delegate, featureMatrix_, model_, maxRules, numThreads_);
                }

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
              uint32 minRules, uint32 maxRules) const override {
                throw std::runtime_error("The rule learner does not support to predict binary labels incrementally");
            }
    };

    template<typename FeatureMatrix>
    static inline std::unique_ptr<IBinaryPredictor> createGfmBinaryPredictor(
      const FeatureMatrix& featureMatrix, const RuleList& model, const LabelVectorSet* labelVectorSet, uint32 numLabels,
      const IProbabilityFunctionFactory& probabilityFunctionFactory, uint32 numThreads) {
        if (!labelVectorSet) {
            throw std::runtime_error(
              "Information about the label vectors that have been encountered in the training data is required for "
              "predicting binary labels, but no such information is provided by the model. Most probably, the model "
              "was intended to use a different prediction method when it has been trained.");
        }

        std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr = probabilityFunctionFactory.create();
        return std::make_unique<GfmBinaryPredictor<FeatureMatrix, RuleList>>(
          featureMatrix, model, *labelVectorSet, numLabels, std::move(probabilityFunctionPtr), numThreads);
    }

    /**
     * Allows to create instances of the type `IBinaryPredictor` that allow to predict whether individual labels of
     * given query examples are relevant or irrelevant by summing up the scores that are provided by the individual
     * rules of an existing rule-based model and transforming them into binary values according to the general F-measure
     * maximizer (GFM).
     */
    class GfmBinaryPredictorFactory final : public IBinaryPredictorFactory {
        private:

            std::unique_ptr<IProbabilityFunctionFactory> probabilityFunctionFactoryPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param probabilityFunctionFactoryPtr An unique pointer to an object of type `IProbabilityFunctionFactory`
             *                                      that allows to create implementations of the transformation function
             *                                      to be used to transform predicted scores into probabilities
             * @param numThreads                    The number of CPU threads to be used to make predictions for
             *                                      different query examples in parallel. Must be at least 1
             */
            GfmBinaryPredictorFactory(std::unique_ptr<IProbabilityFunctionFactory> probabilityFunctionFactoryPtr,
                                      uint32 numThreads)
                : probabilityFunctionFactoryPtr_(std::move(probabilityFunctionFactoryPtr)), numThreads_(numThreads) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IBinaryPredictor> create(const CContiguousConstView<const float32>& featureMatrix,
                                                     const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                     uint32 numLabels) const override {
                return createGfmBinaryPredictor(featureMatrix, model, labelVectorSet, numLabels,
                                                *probabilityFunctionFactoryPtr_, numThreads_);
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IBinaryPredictor> create(const CsrConstView<const float32>& featureMatrix,
                                                     const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                     uint32 numLabels) const override {
                return createGfmBinaryPredictor(featureMatrix, model, labelVectorSet, numLabels,
                                                *probabilityFunctionFactoryPtr_, numThreads_);
            }
    };

    /**
     * An implementation of the type `ISparseBinaryPredictor` that allows to predict whether individual labels of given
     * query examples are relevant or irrelevant by summing up the scores that are provided by the individual rules of
     * an existing rule-based model and transforming them into binary values according to the general F-measure
     * maximizer (GFM).
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class GfmSparseBinaryPredictor final : public ISparseBinaryPredictor {
        private:

            typedef BinarySparsePredictionDispatcher<FeatureMatrix, Model> Dispatcher;

            class Delegate final : public Dispatcher::IPredictionDelegate {
                private:

                    CContiguousView<float64>& scoreMatrix_;

                    BinaryLilMatrix& predictionMatrix_;

                    const LabelVectorSet& labelVectorSet_;

                    const IProbabilityFunction& probabilityFunction_;

                    uint32 maxLabelCardinality_;

                public:

                    Delegate(CContiguousView<float64>& scoreMatrix, BinaryLilMatrix& predictionMatrix,
                             const LabelVectorSet& labelVectorSet, const IProbabilityFunction& probabilityFunction,
                             uint32 maxLabelCardinality)
                        : scoreMatrix_(scoreMatrix), predictionMatrix_(predictionMatrix),
                          labelVectorSet_(labelVectorSet), probabilityFunction_(probabilityFunction),
                          maxLabelCardinality_(maxLabelCardinality) {}

                    uint32 predictForExample(const FeatureMatrix& featureMatrix, const Model& model, uint32 maxRules,
                                             uint32 threadIndex, uint32 exampleIndex,
                                             uint32 predictionIndex) const override {
                        uint32 numLabels = scoreMatrix_.getNumCols();
                        CContiguousView<float64>::value_iterator scoreIterator =
                          scoreMatrix_.row_values_begin(threadIndex);
                        setArrayToZeros(scoreIterator, numLabels);
                        ScorePredictionDelegate<FeatureMatrix, Model>(scoreMatrix_)
                          .predictForExample(featureMatrix, model, maxRules, threadIndex, exampleIndex, threadIndex);
                        BinaryLilMatrix::row predictionRow = predictionMatrix_[predictionIndex];
                        uint32 numLabelVectors = labelVectorSet_.getNumLabelVectors();
                        predictGfm<BinaryLilMatrix::row>(scoreIterator, predictionRow, numLabels, probabilityFunction_,
                                                         labelVectorSet_, numLabelVectors, maxLabelCardinality_);
                        return (uint32) predictionRow.size();
                    }
            };

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            uint32 numLabels_;

            uint32 numThreads_;

            const LabelVectorSet& labelVectorSet_;

            std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr_;

        public:

            /**
             * @param featureMatrix             A reference to an object of template type `FeatureMatrix` that provide
             *                                  row-wise access to the feature values of the query examples
             * @param model                     A reference to an object of template type `Model` that should be used to
             *                                  obtain predictions
             * @param labelVectorSet            A reference to an object of type `LabelVectorSet` that stores all known
             *                                  label vectors
             * @param numLabels                 The number of labels to predict for
             * @param probabilityFunctionPtr    An unique pointer to an object of type `IProbabilityFunction` that
             *                                  should be used to transform predicted scores into probabilities
             * @param numThreads                The number of CPU threads to be used to make predictions for different
             *                                  query examples in parallel. Must be at least 1
             */
            GfmSparseBinaryPredictor(const FeatureMatrix& featureMatrix, const Model& model,
                                     const LabelVectorSet& labelVectorSet, uint32 numLabels,
                                     std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr, uint32 numThreads)
                : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels), numThreads_(numThreads),
                  labelVectorSet_(labelVectorSet), probabilityFunctionPtr_(std::move(probabilityFunctionPtr)) {}

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<BinarySparsePredictionMatrix> predict(uint32 maxRules) const override {
                uint32 numExamples = featureMatrix_.getNumRows();
                BinaryLilMatrix predictionMatrix(numExamples);
                uint32 numNonZeroElements;

                if (labelVectorSet_.getNumLabelVectors() > 0) {
                    uint32 maxLabelCardinality = getMaxLabelCardinality(labelVectorSet_);
                    DenseMatrix<float64> scoreMatrix(numThreads_, numLabels_);
                    Delegate delegate(scoreMatrix, predictionMatrix, labelVectorSet_, *probabilityFunctionPtr_,
                                      maxLabelCardinality);
                    numNonZeroElements = Dispatcher().predict(delegate, featureMatrix_, model_, maxRules, numThreads_);
                } else {
                    numNonZeroElements = 0;
                }

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
              uint32 minRules, uint32 maxRules) const override {
                throw std::runtime_error(
                  "The rule learner does not support to predict sparse binary labels incrementally");
            }
    };

    template<typename FeatureMatrix>
    static inline std::unique_ptr<ISparseBinaryPredictor> createGfmSparseBinaryPredictor(
      const FeatureMatrix& featureMatrix, const RuleList& model, const LabelVectorSet* labelVectorSet, uint32 numLabels,
      const IProbabilityFunctionFactory& probabilityFunctionFactory, uint32 numThreads) {
        if (!labelVectorSet) {
            throw std::runtime_error(
              "Information about the label vectors that have been encountered in the training data is required for "
              "predicting binary labels, but no such information is provided by the model. Most probably, the model "
              "was intended to use a different prediction method when it has been trained.");
        }

        std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr = probabilityFunctionFactory.create();
        return std::make_unique<GfmSparseBinaryPredictor<FeatureMatrix, RuleList>>(
          featureMatrix, model, *labelVectorSet, numLabels, std::move(probabilityFunctionPtr), numThreads);
    }

    /**
     * Allows to create instances of the type `ISparseBinaryPredictor` that allow to predict whether individual labels
     * of given query examples are relevant or irrelevant by summing up the scores that are provided by the individual
     * rules of an existing rule-based model and transforming them into binary values according to the general F-measure
     * maximizer (GFM).
     */
    class GfmSparseBinaryPredictorFactory final : public ISparseBinaryPredictorFactory {
        private:

            std::unique_ptr<IProbabilityFunctionFactory> probabilityFunctionFactoryPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param probabilityFunctionFactoryPtr An unique pointer to an object of type `IProbabilityFunctionFactory`
             *                                      that allows to create implementations of the transformation function
             *                                      to be used to transform predicted scores into probabilities
             * @param numThreads                    The number of CPU threads to be used to make predictions for
             *                                      different query examples in parallel. Must be at least 1
             */
            GfmSparseBinaryPredictorFactory(std::unique_ptr<IProbabilityFunctionFactory> probabilityFunctionFactoryPtr,
                                            uint32 numThreads)
                : probabilityFunctionFactoryPtr_(std::move(probabilityFunctionFactoryPtr)), numThreads_(numThreads) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<ISparseBinaryPredictor> create(const CContiguousConstView<const float32>& featureMatrix,
                                                           const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                           uint32 numLabels) const override {
                return createGfmSparseBinaryPredictor(featureMatrix, model, labelVectorSet, numLabels,
                                                      *probabilityFunctionFactoryPtr_, numThreads_);
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<ISparseBinaryPredictor> create(const CsrConstView<const float32>& featureMatrix,
                                                           const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                           uint32 numLabels) const override {
                return createGfmSparseBinaryPredictor(featureMatrix, model, labelVectorSet, numLabels,
                                                      *probabilityFunctionFactoryPtr_, numThreads_);
            }
    };

    GfmBinaryPredictorConfig::GfmBinaryPredictorConfig(
      const std::unique_ptr<ILossConfig>& lossConfigPtr,
      const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : lossConfigPtr_(std::move(lossConfigPtr)), multiThreadingConfigPtr_(std::move(multiThreadingConfigPtr)) {}

    std::unique_ptr<IBinaryPredictorFactory> GfmBinaryPredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        std::unique_ptr<IProbabilityFunctionFactory> probabilityFunctionFactoryPtr =
          lossConfigPtr_->createProbabilityFunctionFactory();

        if (probabilityFunctionFactoryPtr) {
            uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
            return std::make_unique<GfmBinaryPredictorFactory>(std::move(probabilityFunctionFactoryPtr), numThreads);
        } else {
            return nullptr;
        }
    }

    std::unique_ptr<ISparseBinaryPredictorFactory> GfmBinaryPredictorConfig::createSparsePredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        std::unique_ptr<IProbabilityFunctionFactory> probabilityFunctionFactoryPtr =
          lossConfigPtr_->createProbabilityFunctionFactory();

        if (probabilityFunctionFactoryPtr) {
            uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
            return std::make_unique<GfmSparseBinaryPredictorFactory>(std::move(probabilityFunctionFactoryPtr),
                                                                     numThreads);
        } else {
            return nullptr;
        }
    }

    bool GfmBinaryPredictorConfig::isLabelVectorSetNeeded() const {
        return true;
    }

}
