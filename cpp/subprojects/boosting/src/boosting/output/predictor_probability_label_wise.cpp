#include "boosting/output/predictor_probability_label_wise.hpp"
#include "boosting/math/math.hpp"
#include "common/validation.hpp"
#include "predictor_common.hpp"
#include "omp.h"


namespace boosting {

    /**
     * Allows to transform the score that is predicted for an individual label into a probability by applying the
     * logistic sigmoid function.
     */
    class LogisticFunction final : public IProbabilityFunction {

        public:

            float64 transform(float64 predictedScore) const override {
                return logisticFunction(predictedScore);
            }

    };

    static inline void applyTransformationFunction(CContiguousConstView<float64>::const_iterator originalIterator,
                                                   CContiguousView<float64>::iterator transformedIterator,
                                                   uint32 numElements,
                                                   const IProbabilityFunction& probabilityFunction) {
        for (uint32 i = 0; i < numElements; i++) {
            float64 originalValue = originalIterator[i];
            float64 transformedValue = probabilityFunction.transform(originalValue);
            transformedIterator[i] = transformedValue;
        }
    }

    LabelWiseProbabilityPredictor::LabelWiseProbabilityPredictor(
            std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr, uint32 numThreads)
        : probabilityFunctionPtr_(std::move(probabilityFunctionPtr)), numThreads_(numThreads) {

    }

    void LabelWiseProbabilityPredictor::predict(const CContiguousFeatureMatrix& featureMatrix,
                                                CContiguousView<float64>& predictionMatrix,
                                                const RuleModel& model, const LabelVectorSet* labelVectors) const {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numLabels = predictionMatrix.getNumCols();
        const CContiguousFeatureMatrix* featureMatrixPtr = &featureMatrix;
        CContiguousView<float64>* predictionMatrixPtr = &predictionMatrix;
        const RuleModel* modelPtr = &model;
        const IProbabilityFunction* probabilityFunctionPtr = probabilityFunctionPtr_.get();

        #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(modelPtr) \
        firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) firstprivate(probabilityFunctionPtr) \
        schedule(dynamic) num_threads(numThreads_)
        for (int64 i = 0; i < numExamples; i++) {
            float64* scoreVector = new float64[numLabels] {};

            for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                const Rule& rule = *it;
                applyRule(rule, featureMatrixPtr->row_cbegin(i), featureMatrixPtr->row_cend(i), &scoreVector[0]);
            }

            applyTransformationFunction(&scoreVector[0], predictionMatrixPtr->row_begin(i), numLabels,
                                        *probabilityFunctionPtr);
            delete[] scoreVector;
        }
    }

    void LabelWiseProbabilityPredictor::predict(const CsrFeatureMatrix& featureMatrix,
                                                CContiguousView<float64>& predictionMatrix,
                                                const RuleModel& model, const LabelVectorSet* labelVectors) const {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numLabels = predictionMatrix.getNumCols();
        uint32 numFeatures = featureMatrix.getNumCols();
        const CsrFeatureMatrix* featureMatrixPtr = &featureMatrix;
        CContiguousView<float64>* predictionMatrixPtr = &predictionMatrix;
        const RuleModel* modelPtr = &model;
        const IProbabilityFunction* probabilityFunctionPtr = probabilityFunctionPtr_.get();

        #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(numFeatures) \
        firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) \
        firstprivate(probabilityFunctionPtr) schedule(dynamic) num_threads(numThreads_)
        for (int64 i = 0; i < numExamples; i++) {
            float64* scoreVector = new float64[numLabels] {};
            float32* tmpArray1 = new float32[numFeatures];
            uint32* tmpArray2 = new uint32[numFeatures] {};
            uint32 n = 1;

            for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                const Rule& rule = *it;
                applyRuleCsr(rule, featureMatrixPtr->row_indices_cbegin(i), featureMatrixPtr->row_indices_cend(i),
                             featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                             &scoreVector[0], &tmpArray1[0], &tmpArray2[0], n);
                n++;
            }

            applyTransformationFunction(&scoreVector[0], predictionMatrixPtr->row_begin(i), numLabels,
                                        *probabilityFunctionPtr);
            delete[] scoreVector;
            delete[] tmpArray1;
            delete[] tmpArray2;
        }
    }

    std::unique_ptr<IProbabilityFunction> LogisticFunctionFactory::create() const {
        return std::make_unique<LogisticFunction>();
    }

    LabelWiseProbabilityPredictorFactory::LabelWiseProbabilityPredictorFactory(
            std::unique_ptr<IProbabilityFunctionFactory> probabilityFunctionFactoryPtr, uint32 numThreads)
        : probabilityFunctionFactoryPtr_(std::move(probabilityFunctionFactoryPtr)), numThreads_(numThreads) {
        assertNotNull("probabilityFunctionFactoryPtr", probabilityFunctionFactoryPtr_.get());
        assertGreaterOrEqual<uint32>("numThreads", numThreads, 1);
    }

    std::unique_ptr<IProbabilityPredictor> LabelWiseProbabilityPredictorFactory::create(const RuleModel& model) const {
        std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr = probabilityFunctionFactoryPtr_->create();
        return std::make_unique<LabelWiseProbabilityPredictor>(std::move(probabilityFunctionPtr), numThreads_);
    }

}
