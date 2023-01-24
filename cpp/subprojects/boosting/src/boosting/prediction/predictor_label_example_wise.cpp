#include "boosting/prediction/predictor_label_example_wise.hpp"
#include "predictor_common.hpp"
#include "omp.h"
#include <algorithm>


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
        const LabelVector* closestLabelVector = measureDistance(it, scoresBegin, scoresEnd, measure, minDistance,
                                                                maxCount);
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

    static inline uint32 predictLabelVector(BinaryLilMatrix::row row, const LabelVector& labelVector) {
        uint32 numElements = labelVector.getNumElements();
        LabelVector::const_iterator iterator = labelVector.cbegin();
        row.reserve(numElements);

        for (uint32 i = 0; i < numElements; i++) {
            uint32 labelIndex = iterator[i];
            row.emplace_back(labelIndex);
        }

        return numElements;
    }

    static inline std::unique_ptr<DensePredictionMatrix<uint8>> predictInternally(
            const CContiguousConstView<const float32>& featureMatrix, const RuleList& model,
            const LabelVectorSet& labelVectorSet, uint32 numLabels, const IDistanceMeasure& distanceMeasure,
            uint32 numThreads) {
        uint32 numExamples = featureMatrix.getNumRows();
        std::unique_ptr<DensePredictionMatrix<uint8>> predictionMatrixPtr =
            std::make_unique<DensePredictionMatrix<uint8>>(numExamples, numLabels, true);

        if (labelVectorSet.getNumLabelVectors() > 0) {
            const CContiguousConstView<const float32>* featureMatrixPtr = &featureMatrix;
            CContiguousView<uint8>* predictionMatrixRawPtr = predictionMatrixPtr.get();
            const RuleList* modelPtr = &model;
            const IDistanceMeasure* distanceMeasurePtr = &distanceMeasure;
            const LabelVectorSet* labelVectorSetPtr = &labelVectorSet;

            #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(modelPtr) \
            firstprivate(featureMatrixPtr) firstprivate(predictionMatrixRawPtr) firstprivate(distanceMeasurePtr) \
            firstprivate(labelVectorSetPtr) schedule(dynamic) num_threads(numThreads)
            for (int64 i = 0; i < numExamples; i++) {
                float64* scoreVector = new float64[numLabels] {};
                applyRules(*modelPtr, featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                           &scoreVector[0]);
                const LabelVector& closestLabelVector = findClosestLabelVector(&scoreVector[0], &scoreVector[numLabels],
                                                                               *distanceMeasurePtr, *labelVectorSetPtr);
                predictLabelVector(predictionMatrixRawPtr->row_values_begin(i), closestLabelVector);
                delete[] scoreVector;
            }
        }

        return predictionMatrixPtr;
    }

    static inline std::unique_ptr<DensePredictionMatrix<uint8>> predictInternally(
            const CsrConstView<const float32>& featureMatrix, const RuleList& model,
            const LabelVectorSet& labelVectorSet, uint32 numLabels, const IDistanceMeasure& distanceMeasure,
            uint32 numThreads) {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numFeatures = featureMatrix.getNumCols();
        std::unique_ptr<DensePredictionMatrix<uint8>> predictionMatrixPtr =
            std::make_unique<DensePredictionMatrix<uint8>>(numExamples, numLabels, true);

        if (labelVectorSet.getNumLabelVectors() > 0) {
            const CsrConstView<const float32>* featureMatrixPtr = &featureMatrix;
            CContiguousView<uint8>* predictionMatrixRawPtr = predictionMatrixPtr.get();
            const RuleList* modelPtr = &model;
            const IDistanceMeasure* distanceMeasurePtr = &distanceMeasure;
            const LabelVectorSet* labelVectorSetPtr = &labelVectorSet;

            #pragma omp parallel for firstprivate(numExamples) firstprivate(numFeatures) firstprivate(numLabels) \
            firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixRawPtr) \
            firstprivate(distanceMeasurePtr) firstprivate(labelVectorSetPtr) schedule(dynamic) num_threads(numThreads)
            for (int64 i = 0; i < numExamples; i++) {
                float64* scoreVector = new float64[numLabels] {};
                applyRulesCsr(*modelPtr, numFeatures, featureMatrixPtr->row_indices_cbegin(i),
                              featureMatrixPtr->row_indices_cend(i), featureMatrixPtr->row_values_cbegin(i),
                              featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
                const LabelVector& closestLabelVector = findClosestLabelVector(&scoreVector[0], &scoreVector[numLabels],
                                                                               *distanceMeasurePtr, *labelVectorSetPtr);
                predictLabelVector(predictionMatrixRawPtr->row_values_begin(i), closestLabelVector);
                delete[] scoreVector;
            }
        }

        return predictionMatrixPtr;
    }

    /**
     * An implementation of the type `ILabelPredictor` that allows to predict known label vectors for given query
     * examples by summing up the scores that are provided by an existing rule-based model and comparing the aggregated
     * score vector to the known label vectors according to a certain distance measure. The label vector that is closest
     * to the aggregated score vector is finally predicted.
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class ExampleWiseLabelPredictor final : public ILabelPredictor {

        private:

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            const LabelVectorSet& labelVectorSet_;

            uint32 numLabels_;

            std::unique_ptr<IDistanceMeasure> distanceMeasurePtr_;

            uint32 numThreads_;

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
            ExampleWiseLabelPredictor(const FeatureMatrix& featureMatrix, const Model& model,
                                      const LabelVectorSet& labelVectorSet, uint32 numLabels,
                                      std::unique_ptr<IDistanceMeasure> distanceMeasurePtr, uint32 numThreads)
                : featureMatrix_(featureMatrix), model_(model), labelVectorSet_(labelVectorSet), numLabels_(numLabels),
                  distanceMeasurePtr_(std::move(distanceMeasurePtr)), numThreads_(numThreads) {

            }

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<uint8>> predict() const override {
                return predictInternally(featureMatrix_, model_, labelVectorSet_, numLabels_, *distanceMeasurePtr_,
                                         numThreads_);
            }

    };

    template<typename FeatureMatrix>
    static inline std::unique_ptr<ILabelPredictor> createExampleWiseLabelPredictor(
            const FeatureMatrix& featureMatrix, const RuleList& model, const LabelVectorSet* labelVectorSet,
            uint32 numLabels, const IDistanceMeasureFactory& distanceMeasureFactory, uint32 numThreads) {
        if (!labelVectorSet) {
            throw std::runtime_error("Information about the label vectors that have been encountered in the training "
                "data is required for predicting binary labels, but no such information is provided by the model. Most "
                "probably, the model was intended to use a different prediction method when it has been trained.");
        }

        std::unique_ptr<IDistanceMeasure> distanceMeasurePtr = distanceMeasureFactory.createDistanceMeasure();
        return std::make_unique<ExampleWiseLabelPredictor<FeatureMatrix, RuleList>>(featureMatrix, model,
                                                                                    *labelVectorSet, numLabels,
                                                                                    std::move(distanceMeasurePtr),
                                                                                    numThreads);
    }

    /**
     * Allows to create instances of the type `ILabelPredictor` that allow to predict known label vectors for given
     * query examples by summing up the scores that are provided by an existing rule-based model and comparing the
     * aggregated score vector to the known label vectors according to a certain distance measure. The label vector that
     * is closest to the aggregated score vector is finally predicted.
     */
    class ExampleWiseLabelPredictorFactory final : public ILabelPredictorFactory {

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
            ExampleWiseLabelPredictorFactory(std::unique_ptr<IDistanceMeasureFactory> distanceMeasureFactoryPtr,
                                             uint32 numThreads)
                : distanceMeasureFactoryPtr_(std::move(distanceMeasureFactoryPtr)), numThreads_(numThreads) {

            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<ILabelPredictor> create(const CContiguousConstView<const float32>& featureMatrix,
                                                    const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                    uint32 numLabels) const override {
                return createExampleWiseLabelPredictor(featureMatrix, model, labelVectorSet, numLabels,
                                                       *distanceMeasureFactoryPtr_, numThreads_);
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<ILabelPredictor> create(const CsrConstView<const float32>& featureMatrix,
                                                    const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                    uint32 numLabels) const override {
                return createExampleWiseLabelPredictor(featureMatrix, model, labelVectorSet, numLabels,
                                                       *distanceMeasureFactoryPtr_, numThreads_);
            }

    };

    static inline std::unique_ptr<BinarySparsePredictionMatrix> predictSparseInternally(
            const CContiguousConstView<const float32>& featureMatrix, const RuleList& model,
            const LabelVectorSet& labelVectorSet, uint32 numLabels, const IDistanceMeasure& distanceMeasure,
            uint32 numThreads) {
        uint32 numExamples = featureMatrix.getNumRows();
        BinaryLilMatrix lilMatrix(numExamples);
        uint32 numNonZeroElements = 0;

        if (labelVectorSet.getNumLabelVectors() > 0) {
            const CContiguousConstView<const float32>* featureMatrixPtr = &featureMatrix;
            BinaryLilMatrix* predictionMatrixPtr = &lilMatrix;
            const RuleList* modelPtr = &model;
            const IDistanceMeasure* distanceMeasurePtr = &distanceMeasure;
            const LabelVectorSet* labelVectorSetPtr = &labelVectorSet;

            #pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) firstprivate(numLabels) \
            firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) \
            firstprivate(distanceMeasurePtr) firstprivate(labelVectorSetPtr) schedule(dynamic) num_threads(numThreads)
            for (int64 i = 0; i < numExamples; i++) {
                float64* scoreVector = new float64[numLabels] {};
                applyRules(*modelPtr, featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                           &scoreVector[0]);
                const LabelVector& closestLabelVector = findClosestLabelVector(&scoreVector[0], &scoreVector[numLabels],
                                                                               *distanceMeasurePtr, *labelVectorSetPtr);
                numNonZeroElements += predictLabelVector((*predictionMatrixPtr)[i], closestLabelVector);
                delete[] scoreVector;
            }
        }

        return createBinarySparsePredictionMatrix(lilMatrix, numLabels, numNonZeroElements);
    }

    static inline std::unique_ptr<BinarySparsePredictionMatrix> predictSparseInternally(
            const CsrConstView<const float32>& featureMatrix, const RuleList& model,
            const LabelVectorSet& labelVectorSet, uint32 numLabels, const IDistanceMeasure& distanceMeasure,
            uint32 numThreads) {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numFeatures = featureMatrix.getNumCols();
        BinaryLilMatrix lilMatrix(numExamples);
        uint32 numNonZeroElements = 0;

        if (labelVectorSet.getNumLabelVectors() > 0) {
            const CsrConstView<const float32>* featureMatrixPtr = &featureMatrix;
            BinaryLilMatrix* predictionMatrixPtr = &lilMatrix;
            const RuleList* modelPtr = &model;
            const IDistanceMeasure* distanceMeasurePtr = &distanceMeasure;
            const LabelVectorSet* labelVectorSetPtr = &labelVectorSet;

            #pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) \
            firstprivate(numFeatures) firstprivate(numLabels) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
            firstprivate(predictionMatrixPtr) firstprivate(distanceMeasurePtr) firstprivate(labelVectorSetPtr) \
            schedule(dynamic) num_threads(numThreads)
            for (int64 i = 0; i < numExamples; i++) {
                float64* scoreVector = new float64[numLabels] {};
                applyRulesCsr(*modelPtr, numFeatures, featureMatrixPtr->row_indices_cbegin(i),
                              featureMatrixPtr->row_indices_cend(i), featureMatrixPtr->row_values_cbegin(i),
                              featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
                const LabelVector& closestLabelVector = findClosestLabelVector(&scoreVector[0], &scoreVector[numLabels],
                                                                               *distanceMeasurePtr, *labelVectorSetPtr);
                numNonZeroElements += predictLabelVector((*predictionMatrixPtr)[i], closestLabelVector);
                delete[] scoreVector;
            }
        }

        return createBinarySparsePredictionMatrix(lilMatrix, numLabels, numNonZeroElements);
    }

    /**
     * An implementation of the type `ISparseLabelPredictor` that allows to predict known label vectors for given query
     * examples by summing up the scores that are provided by an existing rule-based model and comparing the aggregated
     * score vector to the known label vectors according to a certain distance measure. The label vector that is closest
     * to the aggregated score vector is finally predicted.
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class ExampleWiseSparseLabelPredictor final : public ISparseLabelPredictor {

        private:

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            const LabelVectorSet& labelVectorSet_;

            uint32 numLabels_;

            std::unique_ptr<IDistanceMeasure> distanceMeasurePtr_;

            uint32 numThreads_;

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
            ExampleWiseSparseLabelPredictor(const FeatureMatrix& featureMatrix, const Model& model,
                                            const LabelVectorSet& labelVectorSet, uint32 numLabels,
                                            std::unique_ptr<IDistanceMeasure> distanceMeasurePtr, uint32 numThreads)
                : featureMatrix_(featureMatrix), model_(model), labelVectorSet_(labelVectorSet), numLabels_(numLabels),
                  distanceMeasurePtr_(std::move(distanceMeasurePtr)), numThreads_(numThreads) {

            }

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<BinarySparsePredictionMatrix> predict() const override {
                return predictSparseInternally(featureMatrix_, model_, labelVectorSet_, numLabels_,
                                               *distanceMeasurePtr_, numThreads_);
            }

    };

    template<typename FeatureMatrix>
    static inline std::unique_ptr<ISparseLabelPredictor> createExampleWiseSparseLabelPredictor(
            const FeatureMatrix& featureMatrix, const RuleList& model, const LabelVectorSet* labelVectorSet,
            uint32 numLabels, const IDistanceMeasureFactory& distanceMeasureFactory, uint32 numThreads) {
        if (!labelVectorSet) {
            throw std::runtime_error("Information about the label vectors that have been encountered in the training "
                "data is required for predicting binary labels, but no such information is provided by the model. Most "
                "probably, the model was intended to use a different prediction method when it has been trained.");
        }

        std::unique_ptr<IDistanceMeasure> distanceMeasurePtr = distanceMeasureFactory.createDistanceMeasure();
        return std::make_unique<ExampleWiseSparseLabelPredictor<FeatureMatrix, RuleList>>(featureMatrix, model,
                                                                                          *labelVectorSet, numLabels,
                                                                                          std::move(distanceMeasurePtr),
                                                                                          numThreads);
    }

    /**
     * Allows to create instances of the type `ISparseLabelPredictor` that allow to predict known label vectors for
     * given query examples by summing up the scores that are provided by an existing rule-based model and comparing the
     * aggregated score vector to the known label vectors according to a certain distance measure. The label vector that
     * is closest to the aggregated score vector is finally predicted.
     */
    class ExampleWiseSparseLabelPredictorFactory final : public ISparseLabelPredictorFactory {

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
            ExampleWiseSparseLabelPredictorFactory(std::unique_ptr<IDistanceMeasureFactory> distanceMeasureFactoryPtr,
                                                   uint32 numThreads)
                : distanceMeasureFactoryPtr_(std::move(distanceMeasureFactoryPtr)), numThreads_(numThreads) {

            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<ISparseLabelPredictor> create(const CContiguousConstView<const float32>& featureMatrix,
                                                             const RuleList& model,
                                                             const LabelVectorSet* labelVectorSet,
                                                             uint32 numLabels) const override {
                return createExampleWiseSparseLabelPredictor(featureMatrix, model, labelVectorSet, numLabels,
                                                             *distanceMeasureFactoryPtr_, numThreads_);
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<ISparseLabelPredictor> create(const CsrConstView<const float32>& featureMatrix,
                                                             const RuleList& model,
                                                             const LabelVectorSet* labelVectorSet,
                                                             uint32 numLabels) const override {
                return createExampleWiseSparseLabelPredictor(featureMatrix, model, labelVectorSet, numLabels,
                                                             *distanceMeasureFactoryPtr_, numThreads_);
            }

    };

    ExampleWiseLabelPredictorConfig::ExampleWiseLabelPredictorConfig(
            const std::unique_ptr<ILossConfig>& lossConfigPtr,
            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : lossConfigPtr_(lossConfigPtr), multiThreadingConfigPtr_(multiThreadingConfigPtr) {

    }

    std::unique_ptr<ILabelPredictorFactory> ExampleWiseLabelPredictorConfig::createPredictorFactory(
            const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        std::unique_ptr<IDistanceMeasureFactory> distanceMeasureFactoryPtr =
            lossConfigPtr_->createDistanceMeasureFactory();
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
        return std::make_unique<ExampleWiseLabelPredictorFactory>(std::move(distanceMeasureFactoryPtr), numThreads);
    }

    std::unique_ptr<ISparseLabelPredictorFactory> ExampleWiseLabelPredictorConfig::createSparsePredictorFactory(
            const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        std::unique_ptr<IDistanceMeasureFactory> distanceMeasureFactoryPtr =
            lossConfigPtr_->createDistanceMeasureFactory();
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
        return std::make_unique<ExampleWiseSparseLabelPredictorFactory>(std::move(distanceMeasureFactoryPtr),
                                                                        numThreads);
    }

    bool ExampleWiseLabelPredictorConfig::isLabelVectorSetNeeded() const {
        return true;
    }

}
