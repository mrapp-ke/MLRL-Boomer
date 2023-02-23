#include "boosting/prediction/predictor_binary_example_wise.hpp"

#include "common/prediction/predictor_common.hpp"
#include "predictor_common.hpp"

#include <algorithm>
#include <stdexcept>

namespace boosting {

    static inline const LabelVector* measureDistance(LabelVectorSet::const_iterator iterator,
                                                     const float64* scoresBegin, const float64* scoresEnd,
                                                     const IDistanceMeasure& measure, float64& distance,
                                                     uint32& count) {
        const auto& entry = *iterator;
        const std::unique_ptr<LabelVector>& labelVectorPtr = entry.first;
        distance = measure.measureDistance(*labelVectorPtr, scoresBegin, scoresEnd);
        count = entry.second;
        return labelVectorPtr.get();
    }

    static inline const LabelVector& findClosestLabelVector(const float64* scoresBegin, const float64* scoresEnd,
                                                            const IDistanceMeasure& measure,
                                                            const LabelVectorSet& labelVectorSet) {
        float64 minDistance;
        uint32 maxCount;
        LabelVectorSet::const_iterator it = labelVectorSet.cbegin();
        const LabelVector* closestLabelVector =
          measureDistance(it, scoresBegin, scoresEnd, measure, minDistance, maxCount);
        it++;

        for (; it != labelVectorSet.cend(); it++) {
            float64 distance;
            uint32 count;
            const LabelVector* labelVector = measureDistance(it, scoresBegin, scoresEnd, measure, distance, count);

            if (distance < minDistance || (distance == minDistance && count > maxCount)) {
                closestLabelVector = labelVector;
                minDistance = distance;
                maxCount = count;
            }
        }

        return *closestLabelVector;
    }

    static inline void predictLabelVector(CContiguousView<uint8>::value_iterator predictionIterator,
                                          const LabelVector& labelVector) {
        uint32 numIndices = labelVector.getNumElements();
        LabelVector::const_iterator indexIterator = labelVector.cbegin();

        for (uint32 i = 0; i < numIndices; i++) {
            uint32 labelIndex = indexIterator[i];
            predictionIterator[labelIndex] = 1;
        }
    }

    static inline void predictLabelVector(BinaryLilMatrix::row row, const LabelVector& labelVector) {
        uint32 numElements = labelVector.getNumElements();
        LabelVector::const_iterator iterator = labelVector.cbegin();
        row.reserve(numElements);

        for (uint32 i = 0; i < numElements; i++) {
            uint32 labelIndex = iterator[i];
            row.emplace_back(labelIndex);
        }
    }

    static inline void predictForExampleInternally(const RuleList& model,
                                                   const CContiguousConstView<const float32>& featureMatrix,
                                                   CContiguousView<uint8>& predictionMatrix, uint32 maxRules,
                                                   uint32 exampleIndex, const LabelVectorSet& labelVectorSet,
                                                   const IDistanceMeasure& distanceMeasure) {
        uint32 numLabels = predictionMatrix.getNumCols();
        float64* scoreVector = new float64[numLabels] {};
        applyRules(model, maxRules, featureMatrix.row_values_cbegin(exampleIndex),
                   featureMatrix.row_values_cend(exampleIndex), &scoreVector[0]);
        const LabelVector& closestLabelVector =
          findClosestLabelVector(&scoreVector[0], &scoreVector[numLabels], distanceMeasure, labelVectorSet);
        predictLabelVector(predictionMatrix.row_values_begin(exampleIndex), closestLabelVector);
        delete[] scoreVector;
    }

    static inline void predictForExampleInternally(const RuleList& model,
                                                   const CsrConstView<const float32>& featureMatrix,
                                                   CContiguousView<uint8>& predictionMatrix, uint32 maxRules,
                                                   uint32 exampleIndex, const LabelVectorSet& labelVectorSet,
                                                   const IDistanceMeasure& distanceMeasure) {
        uint32 numFeatures = featureMatrix.getNumCols();
        uint32 numLabels = predictionMatrix.getNumCols();
        float64* scoreVector = new float64[numLabels] {};
        applyRules(model, maxRules, numFeatures, featureMatrix.row_indices_cbegin(exampleIndex),
                   featureMatrix.row_indices_cend(exampleIndex), featureMatrix.row_values_cbegin(exampleIndex),
                   featureMatrix.row_values_cend(exampleIndex), &scoreVector[0]);
        const LabelVector& closestLabelVector =
          findClosestLabelVector(&scoreVector[0], &scoreVector[numLabels], distanceMeasure, labelVectorSet);
        predictLabelVector(predictionMatrix.row_values_begin(exampleIndex), closestLabelVector);
        delete[] scoreVector;
    }

    /**
     * An implementation of the type `IBinaryPredictor` that allows to predict known label vectors for given query
     * examples by summing up the scores that are provided by an existing rule-based model and comparing the aggregated
     * score vector to the known label vectors according to a certain distance measure. The label vector that is closest
     * to the aggregated score vector is finally predicted.
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class ExampleWiseBinaryPredictor final : public AbstractPredictor<uint8, FeatureMatrix, Model>,
                                             virtual public IBinaryPredictor {
        private:

            const LabelVectorSet& labelVectorSet_;

            std::unique_ptr<IDistanceMeasure> distanceMeasurePtr_;

        protected:

            /**
             * @see `AbstractPredictor::predictForExample`
             */
            void predictForExample(const Model& model, const FeatureMatrix& featureMatrix,
                                   DensePredictionMatrix<uint8>& predictionMatrix, uint32 maxRules,
                                   uint32 exampleIndex) const override {
                if (labelVectorSet_.getNumLabelVectors() > 0) {
                    predictForExampleInternally(model, featureMatrix, predictionMatrix, maxRules, exampleIndex,
                                                labelVectorSet_, *distanceMeasurePtr_);
                }
            }

        public:

            /**
             * @param featureMatrix         A reference to an object of template type `FeatureMatrix` that provides
             *                              row-wise access to the feature values of the query examples
             * @param model                 A reference to an object of template type `Model` that should be used to
             *                              obtain predictions
             * @param labelVectorSet        A reference to an object of type `LabelVectorSet` that stores all known
             *                              label vectors
             * @param numLabels             The number of labels to predict for
             * @param distanceMeasurePtr    An unique pointer to an object of type `IDistanceMeasure` that implements
             *                              the distance measure that should be used to calculate the distance between
             *                              predicted scores and known label vectors
             * @param numThreads            The number of CPU threads to be used to make predictions for different query
             *                              examples in parallel. Must be at least 1
             */
            ExampleWiseBinaryPredictor(const FeatureMatrix& featureMatrix, const Model& model,
                                       const LabelVectorSet& labelVectorSet, uint32 numLabels,
                                       std::unique_ptr<IDistanceMeasure> distanceMeasurePtr, uint32 numThreads)
                : AbstractPredictor<uint8, FeatureMatrix, Model>(featureMatrix, model, numLabels, numThreads, true),
                  labelVectorSet_(labelVectorSet), distanceMeasurePtr_(std::move(distanceMeasurePtr)) {}

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
    static inline std::unique_ptr<IBinaryPredictor> createExampleWiseBinaryPredictor(
      const FeatureMatrix& featureMatrix, const RuleList& model, const LabelVectorSet* labelVectorSet, uint32 numLabels,
      const IDistanceMeasureFactory& distanceMeasureFactory, uint32 numThreads) {
        if (!labelVectorSet) {
            throw std::runtime_error(
              "Information about the label vectors that have been encountered in the training data is required for "
              "predicting binary labels, but no such information is provided by the model. Most probably, the model "
              "was intended to use a different prediction method when it has been trained.");
        }

        std::unique_ptr<IDistanceMeasure> distanceMeasurePtr = distanceMeasureFactory.createDistanceMeasure();
        return std::make_unique<ExampleWiseBinaryPredictor<FeatureMatrix, RuleList>>(
          featureMatrix, model, *labelVectorSet, numLabels, std::move(distanceMeasurePtr), numThreads);
    }

    /**
     * Allows to create instances of the type `IBinaryPredictor` that allow to predict known label vectors for given
     * query examples by summing up the scores that are provided by an existing rule-based model and comparing the
     * aggregated score vector to the known label vectors according to a certain distance measure. The label vector that
     * is closest to the aggregated score vector is finally predicted.
     */
    class ExampleWiseBinaryPredictorFactory final : public IBinaryPredictorFactory {
        private:

            std::unique_ptr<IDistanceMeasureFactory> distanceMeasureFactoryPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param distanceMeasureFactoryPtr An unique pointer to an object of type `IDistanceMeasureFactory` that
             *                                  allows to create implementations of the distance measure that should be
             *                                  used to calculate the distance between predicted scores and known label
             *                                  vectors
             * @param numThreads                The number of CPU threads to be used to make predictions for different
             *                                  query examples in parallel. Must be at least 1
             */
            ExampleWiseBinaryPredictorFactory(std::unique_ptr<IDistanceMeasureFactory> distanceMeasureFactoryPtr,
                                              uint32 numThreads)
                : distanceMeasureFactoryPtr_(std::move(distanceMeasureFactoryPtr)), numThreads_(numThreads) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IBinaryPredictor> create(const CContiguousConstView<const float32>& featureMatrix,
                                                     const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                     uint32 numLabels) const override {
                return createExampleWiseBinaryPredictor(featureMatrix, model, labelVectorSet, numLabels,
                                                        *distanceMeasureFactoryPtr_, numThreads_);
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IBinaryPredictor> create(const CsrConstView<const float32>& featureMatrix,
                                                     const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                     uint32 numLabels) const override {
                return createExampleWiseBinaryPredictor(featureMatrix, model, labelVectorSet, numLabels,
                                                        *distanceMeasureFactoryPtr_, numThreads_);
            }
    };

    static inline void predictForExampleInternally(const RuleList& model,
                                                   const CContiguousConstView<const float32>& featureMatrix,
                                                   BinaryLilMatrix::row& predictionRow, uint32 numLabels,
                                                   uint32 maxRules, uint32 exampleIndex,
                                                   const LabelVectorSet& labelVectorSet,
                                                   const IDistanceMeasure& distanceMeasure) {
        float64* scoreVector = new float64[numLabels] {};
        applyRules(model, maxRules, featureMatrix.row_values_cbegin(exampleIndex),
                   featureMatrix.row_values_cend(exampleIndex), &scoreVector[0]);
        const LabelVector& closestLabelVector =
          findClosestLabelVector(&scoreVector[0], &scoreVector[numLabels], distanceMeasure, labelVectorSet);
        predictLabelVector(predictionRow, closestLabelVector);
        delete[] scoreVector;
    }

    static inline void predictForExampleInternally(const RuleList& model,
                                                   const CsrConstView<const float32>& featureMatrix,
                                                   BinaryLilMatrix::row& predictionRow, uint32 numLabels,
                                                   uint32 maxRules, uint32 exampleIndex,
                                                   const LabelVectorSet& labelVectorSet,
                                                   const IDistanceMeasure& distanceMeasure) {
        uint32 numFeatures = featureMatrix.getNumCols();
        float64* scoreVector = new float64[numLabels] {};
        applyRules(model, maxRules, numFeatures, featureMatrix.row_indices_cbegin(exampleIndex),
                   featureMatrix.row_indices_cend(exampleIndex), featureMatrix.row_values_cbegin(exampleIndex),
                   featureMatrix.row_values_cend(exampleIndex), &scoreVector[0]);
        const LabelVector& closestLabelVector =
          findClosestLabelVector(&scoreVector[0], &scoreVector[numLabels], distanceMeasure, labelVectorSet);
        predictLabelVector(predictionRow, closestLabelVector);
        delete[] scoreVector;
    }

    /**
     * An implementation of the type `ISparseBinaryPredictor` that allows to predict known label vectors for given query
     * examples by summing up the scores that are provided by an existing rule-based model and comparing the aggregated
     * score vector to the known label vectors according to a certain distance measure. The label vector that is closest
     * to the aggregated score vector is finally predicted.
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class ExampleWiseSparseBinaryPredictor final : public AbstractBinarySparsePredictor<FeatureMatrix, Model>,
                                                   virtual public ISparseBinaryPredictor {
        private:

            const LabelVectorSet& labelVectorSet_;

            std::unique_ptr<IDistanceMeasure> distanceMeasurePtr_;

        protected:

            /**
             * @see `AbstractBinarySparsePredictor::predictForExample`
             */
            void predictForExample(const Model& model, const FeatureMatrix& featureMatrix,
                                   BinaryLilMatrix::row& predictionRow, uint32 numLabels, uint32 maxRules,
                                   uint32 exampleIndex) const override {
                if (labelVectorSet_.getNumLabelVectors() > 0) {
                    predictForExampleInternally(model, featureMatrix, predictionRow, numLabels, maxRules, exampleIndex,
                                                labelVectorSet_, *distanceMeasurePtr_);
                }
            }

        public:

            /**
             * @param featureMatrix         A reference to an object of template type `FeatureMatrix` that provides
             *                              row-wise access to the feature values of the query examples
             * @param model                 A reference to an object of template type `Model` that should be used to
             *                              obtain predictions
             * @param labelVectorSet        A reference to an object of type `LabelVectorSet` that stores all known
             *                              label vectors
             * @param numLabels             The number of labels to predict for
             * @param distanceMeasurePtr    An unique pointer to an object of type `IDistanceMeasure` that implements
             *                              the distance measure that should be used to calculate the distance between
             *                              predicted scores and known label vectors
             * @param numThreads            The number of CPU threads to be used to make predictions for different query
             *                              examples in parallel. Must be at least 1
             */
            ExampleWiseSparseBinaryPredictor(const FeatureMatrix& featureMatrix, const Model& model,
                                             const LabelVectorSet& labelVectorSet, uint32 numLabels,
                                             std::unique_ptr<IDistanceMeasure> distanceMeasurePtr, uint32 numThreads)
                : AbstractBinarySparsePredictor<FeatureMatrix, Model>(featureMatrix, model, numLabels, numThreads),
                  labelVectorSet_(labelVectorSet), distanceMeasurePtr_(std::move(distanceMeasurePtr)) {}

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
    static inline std::unique_ptr<ISparseBinaryPredictor> createExampleWiseSparseBinaryPredictor(
      const FeatureMatrix& featureMatrix, const RuleList& model, const LabelVectorSet* labelVectorSet, uint32 numLabels,
      const IDistanceMeasureFactory& distanceMeasureFactory, uint32 numThreads) {
        if (!labelVectorSet) {
            throw std::runtime_error(
              "Information about the label vectors that have been encountered in the training data is required for "
              "predicting binary labels, but no such information is provided by the model. Most probably, the model "
              "was intended to use a different prediction method when it has been trained.");
        }

        std::unique_ptr<IDistanceMeasure> distanceMeasurePtr = distanceMeasureFactory.createDistanceMeasure();
        return std::make_unique<ExampleWiseSparseBinaryPredictor<FeatureMatrix, RuleList>>(
          featureMatrix, model, *labelVectorSet, numLabels, std::move(distanceMeasurePtr), numThreads);
    }

    /**
     * Allows to create instances of the type `ISparseBinaryPredictor` that allow to predict known label vectors for
     * given query examples by summing up the scores that are provided by an existing rule-based model and comparing the
     * aggregated score vector to the known label vectors according to a certain distance measure. The label vector that
     * is closest to the aggregated score vector is finally predicted.
     */
    class ExampleWiseSparseBinaryPredictorFactory final : public ISparseBinaryPredictorFactory {
        private:

            std::unique_ptr<IDistanceMeasureFactory> distanceMeasureFactoryPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param distanceMeasureFactoryPtr An unique pointer to an object of type `IDistanceMeasureFactory` that
             *                                  allows to create implementations of the distance measure that should be
             *                                  used to calculate the distance between predicted scores and known label
             *                                  vectors
             * @param numThreads                The number of CPU threads to be used to make predictions for different
             *                                  query examples in parallel. Must be at least 1
             */
            ExampleWiseSparseBinaryPredictorFactory(std::unique_ptr<IDistanceMeasureFactory> distanceMeasureFactoryPtr,
                                                    uint32 numThreads)
                : distanceMeasureFactoryPtr_(std::move(distanceMeasureFactoryPtr)), numThreads_(numThreads) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<ISparseBinaryPredictor> create(const CContiguousConstView<const float32>& featureMatrix,
                                                           const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                           uint32 numLabels) const override {
                return createExampleWiseSparseBinaryPredictor(featureMatrix, model, labelVectorSet, numLabels,
                                                              *distanceMeasureFactoryPtr_, numThreads_);
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<ISparseBinaryPredictor> create(const CsrConstView<const float32>& featureMatrix,
                                                           const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                           uint32 numLabels) const override {
                return createExampleWiseSparseBinaryPredictor(featureMatrix, model, labelVectorSet, numLabels,
                                                              *distanceMeasureFactoryPtr_, numThreads_);
            }
    };

    ExampleWiseBinaryPredictorConfig::ExampleWiseBinaryPredictorConfig(
      const std::unique_ptr<ILossConfig>& lossConfigPtr,
      const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : lossConfigPtr_(lossConfigPtr), multiThreadingConfigPtr_(multiThreadingConfigPtr) {}

    std::unique_ptr<IBinaryPredictorFactory> ExampleWiseBinaryPredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        std::unique_ptr<IDistanceMeasureFactory> distanceMeasureFactoryPtr =
          lossConfigPtr_->createDistanceMeasureFactory();
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
        return std::make_unique<ExampleWiseBinaryPredictorFactory>(std::move(distanceMeasureFactoryPtr), numThreads);
    }

    std::unique_ptr<ISparseBinaryPredictorFactory> ExampleWiseBinaryPredictorConfig::createSparsePredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        std::unique_ptr<IDistanceMeasureFactory> distanceMeasureFactoryPtr =
          lossConfigPtr_->createDistanceMeasureFactory();
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
        return std::make_unique<ExampleWiseSparseBinaryPredictorFactory>(std::move(distanceMeasureFactoryPtr),
                                                                         numThreads);
    }

    bool ExampleWiseBinaryPredictorConfig::isLabelVectorSetNeeded() const {
        return true;
    }

}
