#include "boosting/prediction/predictor_probability_label_wise.hpp"

#include "boosting/prediction/probability_function.hpp"
#include "omp.h"
#include "predictor_common.hpp"

namespace boosting {

    static inline void applyTransformationFunction(CContiguousConstView<float64>::value_const_iterator originalIterator,
                                                   CContiguousView<float64>::value_iterator transformedIterator,
                                                   uint32 numElements,
                                                   const IProbabilityFunction& probabilityFunction) {
        for (uint32 i = 0; i < numElements; i++) {
            float64 originalValue = originalIterator[i];
            float64 transformedValue = probabilityFunction.transform(originalValue);
            transformedIterator[i] = transformedValue;
        }
    }

    static inline std::unique_ptr<DensePredictionMatrix<float64>> predictInternally(
            const CContiguousConstView<const float32>& featureMatrix, const RuleList& model, uint32 numLabels,
            const IProbabilityFunction& probabilityFunction, uint32 numThreads) {
        uint32 numExamples = featureMatrix.getNumRows();
        std::unique_ptr<DensePredictionMatrix<float64>> predictionMatrixPtr =
            std::make_unique<DensePredictionMatrix<float64>>(numExamples, numLabels);
        const CContiguousConstView<const float32>* featureMatrixPtr = &featureMatrix;
        CContiguousView<float64>* predictionMatrixRawPtr = predictionMatrixPtr.get();
        const RuleList* modelPtr = &model;
        const IProbabilityFunction* probabilityFunctionPtr = &probabilityFunction;

        #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(modelPtr) \
        firstprivate(featureMatrixPtr) firstprivate(predictionMatrixRawPtr) firstprivate(probabilityFunctionPtr) \
        schedule(dynamic) num_threads(numThreads)
        for (int64 i = 0; i < numExamples; i++) {
            float64* scoreVector = new float64[numLabels] {};

            for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                const RuleList::Rule& rule = *it;
                applyRule(rule, featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                          &scoreVector[0]);
                }

                applyTransformationFunction(&scoreVector[0], predictionMatrixRawPtr->row_values_begin(i), numLabels,
                                            *probabilityFunctionPtr);
                delete[] scoreVector;
            }

            return predictionMatrixPtr;
        }

    static inline std::unique_ptr<DensePredictionMatrix<float64>> predictInternally(
            const CsrConstView<const float32>& featureMatrix, const RuleList& model, uint32 numLabels,
            const IProbabilityFunction& probabilityFunction, uint32 numThreads) {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numFeatures = featureMatrix.getNumCols();
        std::unique_ptr<DensePredictionMatrix<float64>> predictionMatrixPtr =
            std::make_unique<DensePredictionMatrix<float64>>(numExamples, numLabels);
        const CsrConstView<const float32>* featureMatrixPtr = &featureMatrix;
        CContiguousView<float64>* predictionMatrixRawPtr = predictionMatrixPtr.get();
        const RuleList* modelPtr = &model;
        const IProbabilityFunction* probabilityFunctionPtr = &probabilityFunction;

        #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(numFeatures) \
        firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixRawPtr) \
        firstprivate(probabilityFunctionPtr) schedule(dynamic) num_threads(numThreads)
        for (int64 i = 0; i < numExamples; i++) {
            float64* scoreVector = new float64[numLabels] {};
            float32* tmpArray1 = new float32[numFeatures];
            uint32* tmpArray2 = new uint32[numFeatures] {};
            uint32 n = 1;

            for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                const RuleList::Rule& rule = *it;
                applyRuleCsr(rule, featureMatrixPtr->row_indices_cbegin(i), featureMatrixPtr->row_indices_cend(i),
                             featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                             &scoreVector[0], &tmpArray1[0], &tmpArray2[0], n);
                n++;
            }

            applyTransformationFunction(&scoreVector[0], predictionMatrixRawPtr->row_values_begin(i), numLabels,
                                        *probabilityFunctionPtr);
            delete[] scoreVector;
            delete[] tmpArray1;
            delete[] tmpArray2;
        }

        return predictionMatrixPtr;
    }

    /**
     * An implementation of the type `IProbabilityPredictor` that allows to predict label-wise probabilities for given
     * query examples, which estimate the chance of individual labels to be relevant, by summing up the scores that are
     * provided by individual rules of an existing rule-based model and transforming the aggregated scores into
     * probabilities in [0, 1] according to a certain transformation function that is applied to each label
     * individually.
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class LabelWiseProbabilityPredictor final : public IProbabilityPredictor {
        private:

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            uint32 numLabels_;

            std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param featureMatrix             A reference to an object of template type `FeatureMatrix` that provides
             *                                  row-wise access to the feature values of the query examples
             * @param model                     A reference to an object of template type `Model` that should be used to
             *                                  obtain predictions
             * @param numLabels                 The number of labels to predict for
             * @param probabilityFunctionPtr    An unique pointer to an object of type `IProbabilityFunction` that
             *                                  should be used to transform predicted scores into probabilities
             * @param numThreads                The number of CPU threads to be used to make predictions for different
             *                                  query examples in parallel. Must be at least 1
             */
            LabelWiseProbabilityPredictor(const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels,
                                          std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr,
                                          uint32 numThreads)
                : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels),
                  probabilityFunctionPtr_(std::move(probabilityFunctionPtr)), numThreads_(numThreads) {}

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<float64>> predict() const override {
                return predictInternally(featureMatrix_, model_, numLabels_, *probabilityFunctionPtr_, numThreads_);
            }
    };

    /**
     * Allows to create instances of the type `IProbabilityPredictor` that allow to predict label-wise probabilities
     * for given query examples, which estimate the chance of individual labels to be relevant, by summing up the scores
     * that are provided by individual rules of an existing rule-based model and transforming the aggregated scores into
     * probabilities in [0, 1] according to a certain transformation function that is applied to each label
     * individually.
     */
    class LabelWiseProbabilityPredictorFactory final : public IProbabilityPredictorFactory {
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
            LabelWiseProbabilityPredictorFactory(
                    std::unique_ptr<IProbabilityFunctionFactory> probabilityFunctionFactoryPtr, uint32 numThreads)
                : probabilityFunctionFactoryPtr_(std::move(probabilityFunctionFactoryPtr)), numThreads_(numThreads) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IProbabilityPredictor> create(const CContiguousConstView<const float32>& featureMatrix,
                                                          const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                          uint32 numLabels) const override {
                std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr = probabilityFunctionFactoryPtr_->create();
                return std::make_unique<LabelWiseProbabilityPredictor<CContiguousConstView<const float32>, RuleList>>(
                    featureMatrix, model, numLabels, std::move(probabilityFunctionPtr), numThreads_);
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IProbabilityPredictor> create(const CsrConstView<const float32>& featureMatrix,
                                                          const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                          uint32 numLabels) const override {
                std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr = probabilityFunctionFactoryPtr_->create();
                return std::make_unique<LabelWiseProbabilityPredictor<CsrConstView<const float32>, RuleList>>(
                    featureMatrix, model, numLabels, std::move(probabilityFunctionPtr), numThreads_);
            }
    };

    LabelWiseProbabilityPredictorConfig::LabelWiseProbabilityPredictorConfig(
            const std::unique_ptr<ILossConfig>& lossConfigPtr,
            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : lossConfigPtr_(lossConfigPtr), multiThreadingConfigPtr_(multiThreadingConfigPtr) {}

    std::unique_ptr<IProbabilityPredictorFactory> LabelWiseProbabilityPredictorConfig::createPredictorFactory(
            const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        std::unique_ptr<IProbabilityFunctionFactory> probabilityFunctionFactoryPtr =
            lossConfigPtr_->createProbabilityFunctionFactory();

        if (probabilityFunctionFactoryPtr) {
            uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
            return std::make_unique<LabelWiseProbabilityPredictorFactory>(
                std::move(probabilityFunctionFactoryPtr), numThreads);
        } else {
            return nullptr;
        }
    }

    bool LabelWiseProbabilityPredictorConfig::isLabelVectorSetNeeded() const {
        return false;
    }

}
