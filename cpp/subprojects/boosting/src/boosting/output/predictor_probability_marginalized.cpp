#include "boosting/output/predictor_probability_marginalized.hpp"
#include "predictor_common.hpp"
#include "omp.h"


namespace boosting {

    static inline float64 measureSimilarity(LabelVectorSet::const_iterator iterator, const float64* scoresBegin,
                                            const float64* scoresEnd, const ISimilarityMeasure& measure) {
        const auto& entry = *iterator;
        const std::unique_ptr<LabelVector>& labelVectorPtr = entry.first;
        return measure.measureSimilarity(*labelVectorPtr, scoresBegin, scoresEnd);
    }

    static inline float64 calculateDistances(const float64* scoresBegin, const float64* scoresEnd, float64* distances,
                                             const ISimilarityMeasure& measure, const LabelVectorSet& labelVectorSet) {
        LabelVectorSet::const_iterator it = labelVectorSet.cbegin();
        float64 minDistance = measureSimilarity(it, scoresBegin, scoresEnd, measure);
        it++;
        uint32 i = 0;

        for (; it != labelVectorSet.cend(); it++) {
            float64 distance = measureSimilarity(it, scoresBegin, scoresEnd, measure);
            distances[i] = distance;

            if (distance < minDistance) {
                minDistance = distance;
            }

            i++;
        }

        return minDistance;
    }

    static inline float64 normalizeDistances(float64* distances, uint32 numDistances, float64 minDistance) {
        float64 sumOfNormalizedDistances = 0;

        for (uint32 i = 0; i < numDistances; i++) {
            float64 normalizedDistance = minDistance / distances[i];
            distances[i] = normalizedDistance;
            sumOfNormalizedDistances += normalizedDistance;
        }

        return sumOfNormalizedDistances;
    }

    static inline void calculateMarginalizedProbabilities(CContiguousView<float64>::value_iterator predictionIterator,
                                                          const float64* normalizedDistances,
                                                          float64 sumOfNormalizedDistances,
                                                          const LabelVectorSet& labelVectorSet) {
        uint32 i = 0;

        for (auto it = labelVectorSet.cbegin(); it != labelVectorSet.cend(); it++) {
            const auto& entry = *it;
            const std::unique_ptr<LabelVector>& labelVectorPtr = entry.first;
            uint32 numRelevantLabels = labelVectorPtr->getNumElements();
            LabelVector::const_iterator labelIndexIterator = labelVectorPtr->cbegin();
            float64 jointProbability = normalizedDistances[i] / sumOfNormalizedDistances;

            for (uint32 j = 0; j < numRelevantLabels; j++) {
                uint32 labelIndex = labelIndexIterator[j];
                predictionIterator[labelIndex] += jointProbability;
            }

            i++;
        }
    }

    static inline void predictMarginalizedProbabilities(const float64* scoresBegin, const float64* scoresEnd,
                                                        CContiguousView<float64>::value_iterator predictionIterator,
                                                        uint32 numElements, const ISimilarityMeasure& measure,
                                                        const LabelVectorSet& labelVectorSet, uint32 numLabelVectors) {
        float64* distances = new float64[numLabelVectors];
        float64 minDistance = calculateDistances(scoresBegin, scoresEnd, distances, measure, labelVectorSet);
        float64 sumOfNormalizedDistances = normalizeDistances(distances, numLabelVectors, minDistance);
        calculateMarginalizedProbabilities(predictionIterator, distances, sumOfNormalizedDistances, labelVectorSet);
        delete[] distances;
    }

    /**
     * An implementation of the type `IProbabilityPredictor` that allows to predict marginalized probabilities for given
     * for given query examples, which estimate the chance of individual labels to be relevant, by summing up the scores
     * that are provided by individual rules of an existing rule-based models and comparing the aggregated score vector
     * to the known label vectors according to a certain distance measure. The probability for an individual label
     * calculates as the sum of the distances that have been obtained for all label vectors, where the respective label
     * is specified to be relevant, divided by the total sum of all distances.
     *
     * @tparam Model The type of the rule-based model that is used to obtain predictions
     */
    template<typename Model>
    class MarginalizedProbabilityPredictor final : public IProbabilityPredictor {

        private:

            const Model& model_;

            const LabelVectorSet* labelVectorSet_;

            std::unique_ptr<ISimilarityMeasure> similarityMeasurePtr_;

            uint32 numThreads_;

        public:

            /**
             * @param model                 A reference to an object of template type `Model` that should be used to
             *                              obtain predictions
             * @param labelVectorSet        A pointer to an object of type `LabelVectorSet` that stores all known label
             *                              vectors or a null pointer, if no such set is available
             * @param similarityMeasurePtr  An unique pointer to an object of type `ISimilarityMeasure` that implements
             *                              the similarity measure that should be used to quantify the similarity
             *                              between predictions and known label vectors
             * @param numThreads            The number of CPU threads to be used to make predictions for different query
             *                              examples in parallel. Must be at least 1
             */
            MarginalizedProbabilityPredictor(const Model& model, const LabelVectorSet* labelVectorSet,
                                             std::unique_ptr<ISimilarityMeasure> similarityMeasurePtr,
                                             uint32 numThreads)
                : model_(model), labelVectorSet_(labelVectorSet),
                  similarityMeasurePtr_(std::move(similarityMeasurePtr)), numThreads_(numThreads) {

            }

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<float64>> predict(
                    const CContiguousConstView<const float32>& featureMatrix, uint32 numLabels) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                std::unique_ptr<DensePredictionMatrix<float64>> predictionMatrixPtr =
                    std::make_unique<DensePredictionMatrix<float64>>(numExamples, numLabels, true);
                const LabelVectorSet* labelVectorSetPtr = labelVectorSet_;

                if (labelVectorSetPtr) {
                    uint32 numLabelVectors = labelVectorSetPtr->getNumLabelVectors();

                    if (numLabelVectors > 0) {
                        const CContiguousConstView<const float32>* featureMatrixPtr = &featureMatrix;
                        CContiguousView<float64>* predictionMatrixRawPtr = predictionMatrixPtr.get();
                        const Model* modelPtr = &model_;
                        const ISimilarityMeasure* similarityMeasureRawPtr = similarityMeasurePtr_.get();

                        #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) \
                        firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixRawPtr) \
                        firstprivate(similarityMeasureRawPtr) firstprivate(labelVectorSetPtr) schedule(dynamic) \
                        num_threads(numThreads_)
                        for (int64 i = 0; i < numExamples; i++) {
                            float64* scoreVector = new float64[numLabels] {};
                            applyRules(*modelPtr, featureMatrixPtr->row_values_cbegin(i),
                                       featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
                            predictMarginalizedProbabilities(&scoreVector[0], &scoreVector[numLabels],
                                                             predictionMatrixRawPtr->row_values_begin(i), numLabels,
                                                             *similarityMeasureRawPtr, *labelVectorSetPtr,
                                                             numLabelVectors);
                            delete[] scoreVector;
                        }
                    }
                }

                return predictionMatrixPtr;
            }

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<float64>> predict(const CsrConstView<const float32>& featureMatrix,
                                                                    uint32 numLabels) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                uint32 numFeatures = featureMatrix.getNumCols();
                std::unique_ptr<DensePredictionMatrix<float64>> predictionMatrixPtr =
                    std::make_unique<DensePredictionMatrix<float64>>(numExamples, numLabels, true);
                const LabelVectorSet* labelVectorSetPtr = labelVectorSet_;

                if (labelVectorSetPtr) {
                    uint32 numLabelVectors = labelVectorSetPtr->getNumLabelVectors();

                    if (numLabelVectors > 0) {
                        const CsrConstView<const float32>* featureMatrixPtr = &featureMatrix;
                        CContiguousView<float64>* predictionMatrixRawPtr = predictionMatrixPtr.get();
                        const Model* modelPtr = &model_;
                        const ISimilarityMeasure* similarityMeasureRawPtr = similarityMeasurePtr_.get();

                        #pragma omp parallel for firstprivate(numExamples) firstprivate(numFeatures) \
                        firstprivate(numLabels) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
                        firstprivate(predictionMatrixRawPtr) firstprivate(similarityMeasureRawPtr) \
                        firstprivate(labelVectorSetPtr) schedule(dynamic) num_threads(numThreads_)
                        for (int64 i = 0; i < numExamples; i++) {
                            float64* scoreVector = new float64[numLabels] {};
                            applyRulesCsr(*modelPtr, numFeatures, featureMatrixPtr->row_indices_cbegin(i),
                                          featureMatrixPtr->row_indices_cend(i), featureMatrixPtr->row_values_cbegin(i),
                                          featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
                            predictMarginalizedProbabilities(&scoreVector[0], &scoreVector[numLabels],
                                                             predictionMatrixRawPtr->row_values_begin(i), numLabels,
                                                             *similarityMeasureRawPtr, *labelVectorSetPtr,
                                                             numLabelVectors);
                            delete[] scoreVector;
                        }
                    }
                }

                return predictionMatrixPtr;
            }

    };

    /**
     * Allows to create instances of the type `IProbabilityPredictor` that allow to predict marginalized probabilities
     * for given query examples, which estimate the chance of individual labels to be relevant, by summing up the scores
     * that are provided by individual rules of an existing rule-based model and comparing the aggregated score vector
     * to the known label vectors according to a certain distance measure. The probability for an individual label
     * calculates as the sum of the distances that have been obtained for all label vectors, where the respective label
     * is specified to be relevant, divided by the total sum of all distances.
     */
    class MarginalizedProbabilityPredictorFactory final : public IProbabilityPredictorFactory {

        private:

            std::unique_ptr<ISimilarityMeasureFactory> similarityMeasureFactoryPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param similarityMeasureFactoryPtr   An unique pointer to an object of type `ISimilarityMeasureFactory`
             *                                      that allows to create implementations of the similarity measure
             *                                      that should be used to quantify the similarity between predictions
             *                                      and known label vectors
             * @param numThreads                    The number of CPU threads to be used to make predictions for
             *                                      different query examples in parallel. Must be at least 1
             */
            MarginalizedProbabilityPredictorFactory(
                    std::unique_ptr<ISimilarityMeasureFactory> similarityMeasureFactoryPtr, uint32 numThreads)
                : similarityMeasureFactoryPtr_(std::move(similarityMeasureFactoryPtr)), numThreads_(numThreads) {

            }

            /**
             * @see `IProbabilityPredictorFactory::create`
             */
            std::unique_ptr<IProbabilityPredictor> create(const RuleList& model,
                                                          const LabelVectorSet* labelVectorSet) const override {
                std::unique_ptr<ISimilarityMeasure> similarityMeasurePtr =
                    similarityMeasureFactoryPtr_->createSimilarityMeasure();
                return std::make_unique<MarginalizedProbabilityPredictor<RuleList>>(model, labelVectorSet,
                                                                                    std::move(similarityMeasurePtr),
                                                                                    numThreads_);
            }

    };

    MarginalizedProbabilityPredictorConfig::MarginalizedProbabilityPredictorConfig(
            const std::unique_ptr<ILossConfig>& lossConfigPtr,
            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : lossConfigPtr_(std::move(lossConfigPtr)), multiThreadingConfigPtr_(std::move(multiThreadingConfigPtr)) {

    }

    std::unique_ptr<IProbabilityPredictorFactory> MarginalizedProbabilityPredictorConfig::createProbabilityPredictorFactory(
            const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
        std::unique_ptr<ISimilarityMeasureFactory> similarityMeasureFactoryPtr =
            lossConfigPtr_->createSimilarityMeasureFactory();
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
        return std::make_unique<MarginalizedProbabilityPredictorFactory>(std::move(similarityMeasureFactoryPtr),
                                                                         numThreads);
    }

    bool MarginalizedProbabilityPredictorConfig::isLabelVectorSetNeeded() const {
        return true;
    }

}
