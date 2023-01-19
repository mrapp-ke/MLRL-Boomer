#include "boosting/prediction/predictor_probability_marginalized.hpp"
#include "common/math/math.hpp"
#include "predictor_common.hpp"
#include "predictor_probability_common.hpp"
#include "omp.h"


namespace boosting {

    static inline void calculateMarginalizedProbabilities(CContiguousView<float64>::value_iterator predictionIterator,
                                                          const float64* jointProbabilities,
                                                          float64 sumOfJointProbabilities,
                                                          const LabelVectorSet& labelVectorSet) {
        uint32 i = 0;

        for (auto it = labelVectorSet.cbegin(); it != labelVectorSet.cend(); it++) {
            const auto& entry = *it;
            const std::unique_ptr<LabelVector>& labelVectorPtr = entry.first;
            uint32 numRelevantLabels = labelVectorPtr->getNumElements();
            LabelVector::const_iterator labelIndexIterator = labelVectorPtr->cbegin();
            float64 normalizedJointProbability = divideOrZero(jointProbabilities[i], sumOfJointProbabilities);

            for (uint32 j = 0; j < numRelevantLabels; j++) {
                uint32 labelIndex = labelIndexIterator[j];
                predictionIterator[labelIndex] += normalizedJointProbability;
            }

            i++;
        }
    }

    static inline void predictMarginalizedProbabilities(const float64* scoresBegin, const float64* scoresEnd,
                                                        CContiguousView<float64>::value_iterator predictionIterator,
                                                        const IProbabilityFunction& probabilityFunction,
                                                        const LabelVectorSet& labelVectorSet, uint32 numLabelVectors) {
        float64* jointProbabilities = new float64[numLabelVectors];
        float64 sumOfJointProbabilities = calculateJointProbabilities(scoresBegin, scoresEnd, jointProbabilities,
                                                                      probabilityFunction, labelVectorSet);
        calculateMarginalizedProbabilities(predictionIterator, jointProbabilities, sumOfJointProbabilities,
                                           labelVectorSet);
        delete[] jointProbabilities;
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
    class MarginalizedProbabilityPredictor final : public IOldProbabilityPredictor {

        private:

            const Model& model_;

            const LabelVectorSet& labelVectorSet_;

            std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param model                     A reference to an object of template type `Model` that should be used to
             *                                  obtain predictions
             * @param labelVectorSet            A reference to an object of type `LabelVectorSet` that stores all known
             *                                  label vectors
             * @param probabilityFunctionPtr    An unique pointer to an object of type `IProbabilityFunction` that
             *                                  should be used to transform predicted scores into probabilities
             * @param numThreads                The number of CPU threads to be used to make predictions for different
             *                                  query examples in parallel. Must be at least 1
             */
            MarginalizedProbabilityPredictor(const Model& model, const LabelVectorSet& labelVectorSet,
                                             std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr,
                                             uint32 numThreads)
                : model_(model), labelVectorSet_(labelVectorSet),
                  probabilityFunctionPtr_(std::move(probabilityFunctionPtr)), numThreads_(numThreads) {

            }

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<float64>> predict(
                    const CContiguousConstView<const float32>& featureMatrix, uint32 numLabels) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                std::unique_ptr<DensePredictionMatrix<float64>> predictionMatrixPtr =
                    std::make_unique<DensePredictionMatrix<float64>>(numExamples, numLabels, true);
                uint32 numLabelVectors = labelVectorSet_.getNumLabelVectors();

                if (numLabelVectors > 0) {
                    const CContiguousConstView<const float32>* featureMatrixPtr = &featureMatrix;
                    CContiguousView<float64>* predictionMatrixRawPtr = predictionMatrixPtr.get();
                    const Model* modelPtr = &model_;
                    const IProbabilityFunction* probabilityFunctionPtr = probabilityFunctionPtr_.get();
                    const LabelVectorSet* labelVectorSetPtr = &labelVectorSet_;

                    #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(modelPtr) \
                    firstprivate(featureMatrixPtr) firstprivate(predictionMatrixRawPtr) \
                    firstprivate(probabilityFunctionPtr) firstprivate(labelVectorSetPtr) firstprivate(numLabelVectors) \
                    schedule(dynamic) num_threads(numThreads_)
                    for (int64 i = 0; i < numExamples; i++) {
                        float64* scoreVector = new float64[numLabels] {};
                        applyRules(*modelPtr, featureMatrixPtr->row_values_cbegin(i),
                                   featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
                        predictMarginalizedProbabilities(&scoreVector[0], &scoreVector[numLabels],
                                                         predictionMatrixRawPtr->row_values_begin(i),
                                                         *probabilityFunctionPtr, *labelVectorSetPtr, numLabelVectors);
                        delete[] scoreVector;
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
                uint32 numLabelVectors = labelVectorSet_.getNumLabelVectors();

                if (numLabelVectors > 0) {
                    const CsrConstView<const float32>* featureMatrixPtr = &featureMatrix;
                    CContiguousView<float64>* predictionMatrixRawPtr = predictionMatrixPtr.get();
                    const Model* modelPtr = &model_;
                    const IProbabilityFunction* probabilityFunctionPtr = probabilityFunctionPtr_.get();
                    const LabelVectorSet* labelVectorSetPtr = &labelVectorSet_;

                    #pragma omp parallel for firstprivate(numExamples) firstprivate(numFeatures) \
                    firstprivate(numLabels) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
                    firstprivate(predictionMatrixRawPtr) firstprivate(probabilityFunctionPtr) \
                    firstprivate(labelVectorSetPtr) firstprivate(numLabelVectors) schedule(dynamic) \
                    num_threads(numThreads_)
                    for (int64 i = 0; i < numExamples; i++) {
                        float64* scoreVector = new float64[numLabels] {};
                        applyRulesCsr(*modelPtr, numFeatures, featureMatrixPtr->row_indices_cbegin(i),
                                      featureMatrixPtr->row_indices_cend(i), featureMatrixPtr->row_values_cbegin(i),
                                      featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
                        predictMarginalizedProbabilities(&scoreVector[0], &scoreVector[numLabels],
                                                         predictionMatrixRawPtr->row_values_begin(i),
                                                         *probabilityFunctionPtr, *labelVectorSetPtr, numLabelVectors);
                        delete[] scoreVector;
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
    class MarginalizedProbabilityPredictorFactory final : public IOldProbabilityPredictorFactory {

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
            MarginalizedProbabilityPredictorFactory(
                    std::unique_ptr<IProbabilityFunctionFactory> probabilityFunctionFactoryPtr, uint32 numThreads)
                : probabilityFunctionFactoryPtr_(std::move(probabilityFunctionFactoryPtr)), numThreads_(numThreads) {

            }

            /**
             * @see `IProbabilityPredictorFactory::create`
             */
            std::unique_ptr<IOldProbabilityPredictor> create(const RuleList& model,
                                                             const LabelVectorSet* labelVectorSet) const override {
                if (!labelVectorSet) {
                    throw std::runtime_error("Information about the label vectors that have been encountered in the "
                        "training data is required for predicting binary labels, but no such information is provided "
                        "by the model. Most probably, the model was intended to use a different prediction method when "
                        "it has been trained.");
                }

                std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr = probabilityFunctionFactoryPtr_->create();
                return std::make_unique<MarginalizedProbabilityPredictor<RuleList>>(model, *labelVectorSet,
                                                                                    std::move(probabilityFunctionPtr),
                                                                                    numThreads_);
            }

    };

    MarginalizedProbabilityPredictorConfig::MarginalizedProbabilityPredictorConfig(
            const std::unique_ptr<ILossConfig>& lossConfigPtr,
            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : lossConfigPtr_(std::move(lossConfigPtr)), multiThreadingConfigPtr_(std::move(multiThreadingConfigPtr)) {

    }

    std::unique_ptr<IOldProbabilityPredictorFactory> MarginalizedProbabilityPredictorConfig::createProbabilityPredictorFactory(
            const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
        std::unique_ptr<IProbabilityFunctionFactory> probabilityFunctionFactoryPtr =
            lossConfigPtr_->createProbabilityFunctionFactory();

        if (probabilityFunctionFactoryPtr) {
            uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
            return std::make_unique<MarginalizedProbabilityPredictorFactory>(std::move(probabilityFunctionFactoryPtr),
                                                                             numThreads);
        } else {
            return nullptr;
        }
    }

    bool MarginalizedProbabilityPredictorConfig::isLabelVectorSetNeeded() const {
        return true;
    }

}
