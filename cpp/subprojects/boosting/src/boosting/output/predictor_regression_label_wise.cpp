#include "boosting/output/predictor_regression_label_wise.hpp"
#include "common/validation.hpp"
#include "predictor_common.hpp"
#include "omp.h"


namespace boosting {

    /**
     * An implementation of the type `IRegressionPredictor` that allows to predict label-wise regression scores for
     * given query examples by summing up the scores that are provided by the individual rules of an existing rule-based
     * model for each label individually.
     *
     * @tparam Model The type of the rule-based model that is used to obtain predictions
     */
    template<typename Model>
    class LabelWiseRegressionPredictor final : public IRegressionPredictor {

        private:

            const Model& model_;

            uint32 numThreads_;

        public:

            /**
             * @param model         A reference to an object of template type `Model` that should be used to obtain
             *                      predictions
             * @param numThreads    The number of CPU threads to be used to make predictions for different query
             *                      examples in parallel. Must be at least 1
             */
            LabelWiseRegressionPredictor(const Model& model, uint32 numThreads)
                : model_(model), numThreads_(numThreads) {

            }

            void predict(const CContiguousFeatureMatrix& featureMatrix, CContiguousView<float64>& predictionMatrix,
                         const LabelVectorSet* labelVectors) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                const CContiguousFeatureMatrix* featureMatrixPtr = &featureMatrix;
                CContiguousView<float64>* predictionMatrixPtr = &predictionMatrix;
                const Model* modelPtr = &model_;

                #pragma omp parallel for firstprivate(numExamples) firstprivate(modelPtr) \
                firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) schedule(dynamic) \
                num_threads(numThreads_)
                for (int64 i = 0; i < numExamples; i++) {
                    for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                        const Rule& rule = *it;
                        applyRule(rule, featureMatrixPtr->row_cbegin(i), featureMatrixPtr->row_cend(i),
                                  predictionMatrixPtr->row_begin(i));
                    }
                }
            }

            void predict(const CsrFeatureMatrix& featureMatrix, CContiguousView<float64>& predictionMatrix,
                         const LabelVectorSet* labelVectors) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                uint32 numFeatures = featureMatrix.getNumCols();
                const CsrFeatureMatrix* featureMatrixPtr = &featureMatrix;
                CContiguousView<float64>* predictionMatrixPtr = &predictionMatrix;
                const Model* modelPtr = &model_;

                #pragma omp parallel for firstprivate(numExamples) firstprivate(modelPtr) \
                firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) schedule(dynamic) \
                num_threads(numThreads_)
                for (int64 i = 0; i < numExamples; i++) {
                    float32* tmpArray1 = new float32[numFeatures];
                    uint32* tmpArray2 = new uint32[numFeatures] {};
                    uint32 n = 1;

                    for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                        const Rule& rule = *it;
                        applyRuleCsr(rule, featureMatrixPtr->row_indices_cbegin(i),
                                     featureMatrixPtr->row_indices_cend(i), featureMatrixPtr->row_values_cbegin(i),
                                     featureMatrixPtr->row_values_cend(i), predictionMatrixPtr->row_begin(i),
                                     &tmpArray1[0], &tmpArray2[0], n);
                        n++;
                    }

                    delete[] tmpArray1;
                    delete[] tmpArray2;
                }
            }

    };

    LabelWiseRegressionPredictorFactory::LabelWiseRegressionPredictorFactory(uint32 numThreads)
        : numThreads_(numThreads) {
        assertGreaterOrEqual<uint32>("numThreads", numThreads, 1);
    }

    std::unique_ptr<IRegressionPredictor> LabelWiseRegressionPredictorFactory::create(const RuleList& model) const {
        return std::make_unique<LabelWiseRegressionPredictor<RuleList>>(model, numThreads_);
    }

}
