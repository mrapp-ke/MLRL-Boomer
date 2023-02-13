#include "boosting/prediction/predictor_probability_marginalized.hpp"

#include "common/math/math.hpp"
#include "omp.h"
#include "predictor_common.hpp"
#include "predictor_probability_common.hpp"

#include <stdexcept>

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
        float64 sumOfJointProbabilities =
          calculateJointProbabilities(scoresBegin, scoresEnd, jointProbabilities, probabilityFunction, labelVectorSet);
        calculateMarginalizedProbabilities(predictionIterator, jointProbabilities, sumOfJointProbabilities,
                                           labelVectorSet);
        delete[] jointProbabilities;
    }

    static inline std::unique_ptr<DensePredictionMatrix<float64>> predictInternally(
      const CContiguousConstView<const float32>& featureMatrix, const RuleList& model,
      const LabelVectorSet& labelVectorSet, uint32 numLabels, const IProbabilityFunction& probabilityFunction,
      uint32 numThreads, uint32 maxRules) {
        uint32 numExamples = featureMatrix.getNumRows();
        std::unique_ptr<DensePredictionMatrix<float64>> predictionMatrixPtr =
          std::make_unique<DensePredictionMatrix<float64>>(numExamples, numLabels, true);
        uint32 numLabelVectors = labelVectorSet.getNumLabelVectors();

        if (numLabelVectors > 0) {
            const CContiguousConstView<const float32>* featureMatrixPtr = &featureMatrix;
            CContiguousView<float64>* predictionMatrixRawPtr = predictionMatrixPtr.get();
            const RuleList* modelPtr = &model;
            const IProbabilityFunction* probabilityFunctionPtr = &probabilityFunction;
            const LabelVectorSet* labelVectorSetPtr = &labelVectorSet;

#pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(modelPtr) \
  firstprivate(featureMatrixPtr) firstprivate(predictionMatrixRawPtr) firstprivate(probabilityFunctionPtr) \
    firstprivate(labelVectorSetPtr) firstprivate(numLabelVectors) firstprivate(maxRules) schedule(dynamic) \
      num_threads(numThreads)
            for (int64 i = 0; i < numExamples; i++) {
                float64* scoreVector = new float64[numLabels] {};
                applyRules(*modelPtr, maxRules, featureMatrixPtr->row_values_cbegin(i),
                           featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
                predictMarginalizedProbabilities(&scoreVector[0], &scoreVector[numLabels],
                                                 predictionMatrixRawPtr->row_values_begin(i), *probabilityFunctionPtr,
                                                 *labelVectorSetPtr, numLabelVectors);
                delete[] scoreVector;
            }
        }

        return predictionMatrixPtr;
    }

    static inline std::unique_ptr<DensePredictionMatrix<float64>> predictInternally(
      const CsrConstView<const float32>& featureMatrix, const RuleList& model, const LabelVectorSet& labelVectorSet,
      uint32 numLabels, const IProbabilityFunction& probabilityFunction, uint32 numThreads, uint32 maxRules) {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numFeatures = featureMatrix.getNumCols();
        std::unique_ptr<DensePredictionMatrix<float64>> predictionMatrixPtr =
          std::make_unique<DensePredictionMatrix<float64>>(numExamples, numLabels, true);
        uint32 numLabelVectors = labelVectorSet.getNumLabelVectors();

        if (numLabelVectors > 0) {
            const CsrConstView<const float32>* featureMatrixPtr = &featureMatrix;
            CContiguousView<float64>* predictionMatrixRawPtr = predictionMatrixPtr.get();
            const RuleList* modelPtr = &model;
            const IProbabilityFunction* probabilityFunctionPtr = &probabilityFunction;
            const LabelVectorSet* labelVectorSetPtr = &labelVectorSet;

#pragma omp parallel for firstprivate(numExamples) firstprivate(numFeatures) firstprivate(numLabels) \
  firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixRawPtr) \
    firstprivate(probabilityFunctionPtr) firstprivate(labelVectorSetPtr) firstprivate(numLabelVectors) \
      firstprivate(maxRules) schedule(dynamic) num_threads(numThreads)
            for (int64 i = 0; i < numExamples; i++) {
                float64* scoreVector = new float64[numLabels] {};
                applyRulesCsr(*modelPtr, maxRules, numFeatures, featureMatrixPtr->row_indices_cbegin(i),
                              featureMatrixPtr->row_indices_cend(i), featureMatrixPtr->row_values_cbegin(i),
                              featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
                predictMarginalizedProbabilities(&scoreVector[0], &scoreVector[numLabels],
                                                 predictionMatrixRawPtr->row_values_begin(i), *probabilityFunctionPtr,
                                                 *labelVectorSetPtr, numLabelVectors);
                delete[] scoreVector;
            }
        }

        return predictionMatrixPtr;
    }

    /**
     * An implementation of the type `IProbabilityPredictor` that allows to predict marginalized probabilities for given
     * for given query examples, which estimate the chance of individual labels to be relevant, by summing up the scores
     * that are provided by individual rules of an existing rule-based models and comparing the aggregated score vector
     * to the known label vectors according to a certain distance measure. The probability for an individual label
     * calculates as the sum of the distances that have been obtained for all label vectors, where the respective label
     * is specified to be relevant, divided by the total sum of all distances.
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class MarginalizedProbabilityPredictor final : public IProbabilityPredictor {
        private:

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            const LabelVectorSet& labelVectorSet_;

            uint32 numLabels_;

            std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param featureMatrix             A reference to an object of template type `FeatureMatrix` that provides
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
            MarginalizedProbabilityPredictor(const FeatureMatrix& featureMatrix, const Model& model,
                                             const LabelVectorSet& labelVectorSet, uint32 numLabels,
                                             std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr,
                                             uint32 numThreads)
                : featureMatrix_(featureMatrix), model_(model), labelVectorSet_(labelVectorSet), numLabels_(numLabels),
                  probabilityFunctionPtr_(std::move(probabilityFunctionPtr)), numThreads_(numThreads) {}

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<float64>> predict(uint32 maxRules) const override {
                return predictInternally(featureMatrix_, model_, labelVectorSet_, numLabels_, *probabilityFunctionPtr_,
                                         numThreads_, maxRules);
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
            std::unique_ptr<IIncrementalPredictor<DensePredictionMatrix<float64>>> createIncrementalPredictor()
              const override {
                throw std::runtime_error(
                  "The rule learner does not support to predict probability estimates incrementally");
            }
    };

    template<typename FeatureMatrix>
    static inline std::unique_ptr<IProbabilityPredictor> createMarginalizedProbabilityPredictor(
      const FeatureMatrix& featureMatrix, const RuleList& model, const LabelVectorSet* labelVectorSet, uint32 numLabels,
      const IProbabilityFunctionFactory& probabilityFunctionFactory, uint32 numThreads) {
        if (!labelVectorSet) {
            throw std::runtime_error(
              "Information about the label vectors that have been encountered in the training data is required for "
              "predicting binary labels, but no such information is provided by the model. Most probably, the model "
              "was intended to use a different prediction method when it has been trained.");
        }

        std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr = probabilityFunctionFactory.create();
        return std::make_unique<MarginalizedProbabilityPredictor<FeatureMatrix, RuleList>>(
          featureMatrix, model, *labelVectorSet, numLabels, std::move(probabilityFunctionPtr), numThreads);
    }

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
                : probabilityFunctionFactoryPtr_(std::move(probabilityFunctionFactoryPtr)), numThreads_(numThreads) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IProbabilityPredictor> create(const CContiguousConstView<const float32>& featureMatrix,
                                                          const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                          uint32 numLabels) const override {
                return createMarginalizedProbabilityPredictor(featureMatrix, model, labelVectorSet, numLabels,
                                                              *probabilityFunctionFactoryPtr_, numThreads_);
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IProbabilityPredictor> create(const CsrConstView<const float32>& featureMatrix,
                                                          const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                          uint32 numLabels) const override {
                return createMarginalizedProbabilityPredictor(featureMatrix, model, labelVectorSet, numLabels,
                                                              *probabilityFunctionFactoryPtr_, numThreads_);
            }
    };

    MarginalizedProbabilityPredictorConfig::MarginalizedProbabilityPredictorConfig(
      const std::unique_ptr<ILossConfig>& lossConfigPtr,
      const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : lossConfigPtr_(std::move(lossConfigPtr)), multiThreadingConfigPtr_(std::move(multiThreadingConfigPtr)) {}

    std::unique_ptr<IProbabilityPredictorFactory> MarginalizedProbabilityPredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
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
