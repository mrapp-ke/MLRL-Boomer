#include "boosting/prediction/predictor_score_label_wise.hpp"
#include "predictor_common.hpp"
#include "omp.h"


namespace boosting {

    template<typename Model>
    static inline std::unique_ptr<DensePredictionMatrix<float64>> predictInternally(
            const CContiguousConstView<const float32>& featureMatrix, const Model& model, uint32 numLabels,
            uint32 numThreads) {
        uint32 numExamples = featureMatrix.getNumRows();
        std::unique_ptr<DensePredictionMatrix<float64>> predictionMatrixPtr =
            std::make_unique<DensePredictionMatrix<float64>>(numExamples, numLabels, true);
        const CContiguousConstView<const float32>* featureMatrixPtr = &featureMatrix;
        CContiguousView<float64>* predictionMatrixRawPtr = predictionMatrixPtr.get();
        const Model* modelPtr = &model;

        #pragma omp parallel for firstprivate(numExamples) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
        firstprivate(predictionMatrixRawPtr) schedule(dynamic) num_threads(numThreads)
        for (int64 i = 0; i < numExamples; i++) {
            for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                const RuleList::Rule& rule = *it;
                applyRule(rule, featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                          predictionMatrixRawPtr->row_values_begin(i));
            }
        }

        return predictionMatrixPtr;
    }

    template<typename Model>
    static inline std::unique_ptr<DensePredictionMatrix<float64>> predictInternally(
            const CsrConstView<const float32>& featureMatrix, const Model& model, uint32 numLabels, uint32 numThreads) {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numFeatures = featureMatrix.getNumCols();
        std::unique_ptr<DensePredictionMatrix<float64>> predictionMatrixPtr =
            std::make_unique<DensePredictionMatrix<float64>>(numExamples, numLabels, true);
        const CsrConstView<const float32>* featureMatrixPtr = &featureMatrix;
        CContiguousView<float64>* predictionMatrixRawPtr = predictionMatrixPtr.get();
        const Model* modelPtr = &model;

        #pragma omp parallel for firstprivate(numExamples) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
        firstprivate(predictionMatrixRawPtr) schedule(dynamic) num_threads(numThreads)
        for (int64 i = 0; i < numExamples; i++) {
            float32* tmpArray1 = new float32[numFeatures];
            uint32* tmpArray2 = new uint32[numFeatures] {};
            uint32 n = 1;

            for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                const RuleList::Rule& rule = *it;
                applyRuleCsr(rule, featureMatrixPtr->row_indices_cbegin(i), featureMatrixPtr->row_indices_cend(i),
                             featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                             predictionMatrixRawPtr->row_values_begin(i), &tmpArray1[0], &tmpArray2[0], n);
                n++;
            }

            delete[] tmpArray1;
            delete[] tmpArray2;
        }

        return predictionMatrixPtr;
    }

    /**
     * An implementation of the type `IScorePredictor` that allows to predict label-wise regression scores for given
     * query examples by summing up the scores that are provided by the individual rules of an existing rule-based model
     * for each label individually.
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class LabelWiseScorePredictor final : public IScorePredictor {

        private:

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            uint32 numLabels_;

            uint32 numThreads_;

        public:

            /**
             * @param featureMatrix A reference to an object of template type `FeatureMatrix` that provides row-wise
             *                      access to the feature values of the query examples
             * @param model         A reference to an object of template type `Model` that should be used to obtain
             *                      predictions
             * @param numLabels     The number of labels to predict for
             * @param numThreads    The number of CPU threads to be used to make predictions for different query
             *                      examples in parallel. Must be at least 1
             */
            LabelWiseScorePredictor(const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels,
                                    uint32 numThreads)
                : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels), numThreads_(numThreads) {

            }

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<float64>> predict() const override {
                return predictInternally(featureMatrix_, model_, numLabels_, numThreads_);
            }

    };

    /**
     * Allows to create instances of the type `IScorePredictor` that allow to predict label-wise regression scores for
     * given query examples by summing up the scores that are provided by the individual rules of an existing rule-based
     * model for each label individually.
     */
    class LabelWiseScorePredictorFactory final : public IScorePredictorFactory {

        private:

            uint32 numThreads_;

        public:

            /**
             * @param numThreads The number of CPU threads to be used to make predictions for different query examples
             *                   in parallel. Must be at least 1
             */
            LabelWiseScorePredictorFactory(uint32 numThreads)
                : numThreads_(numThreads) {

            }

            /**
             * @see `IScorePredictorFactory::create`
             */
            std::unique_ptr<IScorePredictor> create(const CContiguousConstView<const float32>& featureMatrix,
                                                    const RuleList& model,
                                                    const LabelVectorSet* labelVectorSet,
                                                    uint32 numLabels) const override {
                return std::make_unique<LabelWiseScorePredictor<CContiguousConstView<const float32>, RuleList>>(
                    featureMatrix, model, numLabels, numThreads_);
            }

            /**
             * @see `IScorePredictorFactory::create`
             */
            std::unique_ptr<IScorePredictor> create(const CsrConstView<const float32>& featureMatrix,
                                                    const RuleList& model,
                                                    const LabelVectorSet* labelVectorSet,
                                                    uint32 numLabels) const override {
                return std::make_unique<LabelWiseScorePredictor<CsrConstView<const float32>, RuleList>>(
                    featureMatrix, model, numLabels, numThreads_);
            }

    };

    LabelWiseScorePredictorConfig::LabelWiseScorePredictorConfig(
            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : multiThreadingConfigPtr_(multiThreadingConfigPtr) {

    }

    std::unique_ptr<IScorePredictorFactory> LabelWiseScorePredictorConfig::createPredictorFactory(
            const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
        return std::make_unique<LabelWiseScorePredictorFactory>(numThreads);
    }

    bool LabelWiseScorePredictorConfig::isLabelVectorSetNeeded() const {
        return false;
    }

}
