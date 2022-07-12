#include "boosting/output/predictor_classification_gfm.hpp"
#include "predictor_common.hpp"
#include "omp.h"


namespace boosting {

    /**
     * An implementation of the type `IClassificationPredictor` that allows to predict whether individual labels of
     * given query examples are relevant or irrelevant by summing up the scores that are provided by the individual
     * rules of an existing rule-based model and transforming them into binary values according to the general F-measure
     * maximizer (GFM).
     *
     * @tparam Model The type of the rule-based model that is used to obtain predictions
     */
    template<typename Model>
    class GfmClassificationPredictor final : public IClassificationPredictor {

        private:

            const Model& model_;

            const LabelVectorSet* labelVectorSet_;

            std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param model                     A reference to an object of template type `Model` that should be used to
             *                                  obtain predictions
             * @param labelVectorSet            A pointer to an object of type `LabelVectorSet` that stores all known
             *                                  label vectors or a null pointer, if no such set is available
             * @param probabilityFunctionPtr    An unique pointer to an object of type `IProbabilityFunction` that
             *                                  should be used to transform predicted scores into probabilities
             * @param numThreads                The number of CPU threads to be used to make predictions for different
             *                                  query examples in parallel. Must be at least 1
             */
            GfmClassificationPredictor(const Model& model, const LabelVectorSet* labelVectorSet,
                                       std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr, uint32 numThreads)
                : model_(model), labelVectorSet_(labelVectorSet),
                  probabilityFunctionPtr_(std::move(probabilityFunctionPtr)), numThreads_(numThreads) {

            }

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<uint8>> predict(
                    const CContiguousConstView<const float32>& featureMatrix, uint32 numLabels) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                std::unique_ptr<DensePredictionMatrix<uint8>> predictionMatrixPtr =
                    std::make_unique<DensePredictionMatrix<uint8>>(numExamples, numLabels, true);
                const LabelVectorSet* labelVectorSetPtr = labelVectorSet_;

                if (labelVectorSetPtr && labelVectorSetPtr->getNumLabelVectors() > 0) {
                    const CContiguousConstView<const float32>* featureMatrixPtr = &featureMatrix;
                    CContiguousView<uint8>* predictionMatrixRawPtr = predictionMatrixPtr.get();
                    const Model* modelPtr = &model_;
                    const IProbabilityFunction* probabilityFunctionPtr = probabilityFunctionPtr_.get();

                    #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(modelPtr) \
                    firstprivate(featureMatrixPtr) firstprivate(predictionMatrixRawPtr) \
                    firstprivate(probabilityFunctionPtr) firstprivate(labelVectorSetPtr) schedule(dynamic) \
                    num_threads(numThreads_)
                    for (int64 i = 0; i < numExamples; i++) {
                        float64* scoreVector = new float64[numLabels] {};
                        applyRules(*modelPtr, featureMatrixPtr->row_values_cbegin(i),
                                   featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
                        // TODO
                        delete[] scoreVector;
                    }
                }

                return predictionMatrixPtr;
            }

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<uint8>> predict(const CsrConstView<const float32>& featureMatrix,
                                                                  uint32 numLabels) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                uint32 numFeatures = featureMatrix.getNumCols();
                std::unique_ptr<DensePredictionMatrix<uint8>> predictionMatrixPtr =
                    std::make_unique<DensePredictionMatrix<uint8>>(numExamples, numLabels, true);
                const LabelVectorSet* labelVectorSetPtr = labelVectorSet_;

                if (labelVectorSetPtr && labelVectorSetPtr->getNumLabelVectors() > 0) {
                    const CsrConstView<const float32>* featureMatrixPtr = &featureMatrix;
                    CContiguousView<uint8>* predictionMatrixRawPtr = predictionMatrixPtr.get();
                    const Model* modelPtr = &model_;
                    const IProbabilityFunction* probabilityFunctionPtr = probabilityFunctionPtr_.get();

                    #pragma omp parallel for firstprivate(numExamples) firstprivate(numFeatures) \
                    firstprivate(numLabels) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
                    firstprivate(predictionMatrixRawPtr) firstprivate(probabilityFunctionPtr) \
                    firstprivate(labelVectorSetPtr) schedule(dynamic) num_threads(numThreads_)
                    for (int64 i = 0; i < numExamples; i++) {
                        float64* scoreVector = new float64[numLabels] {};
                        applyRulesCsr(*modelPtr, numFeatures, featureMatrixPtr->row_indices_cbegin(i),
                                      featureMatrixPtr->row_indices_cend(i), featureMatrixPtr->row_values_cbegin(i),
                                      featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
                        // TODO
                        delete[] scoreVector;
                    }
                }

                return predictionMatrixPtr;
            }

            /**
             * @see `ISparsePredictor::predictSparse`
             */
            std::unique_ptr<BinarySparsePredictionMatrix> predictSparse(
                    const CContiguousConstView<const float32>& featureMatrix, uint32 numLabels) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                BinaryLilMatrix lilMatrix(numExamples);
                uint32 numNonZeroElements = 0;
                const LabelVectorSet* labelVectorSetPtr = labelVectorSet_;

                if (labelVectorSetPtr && labelVectorSetPtr->getNumLabelVectors() > 0) {
                    const CContiguousConstView<const float32>* featureMatrixPtr = &featureMatrix;
                    BinaryLilMatrix* predictionMatrixPtr = &lilMatrix;
                    const Model* modelPtr = &model_;
                    const IProbabilityFunction* probabilityFunctionPtr = probabilityFunctionPtr_.get();

                    #pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) \
                    firstprivate(numLabels) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
                    firstprivate(predictionMatrixPtr) firstprivate(probabilityFunctionPtr) \
                    firstprivate(labelVectorSetPtr) schedule(dynamic) num_threads(numThreads_)
                    for (int64 i = 0; i < numExamples; i++) {
                        float64* scoreVector = new float64[numLabels] {};
                        applyRules(*modelPtr, featureMatrixPtr->row_values_cbegin(i),
                                   featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
                        // TODO
                        delete[] scoreVector;
                    }
                }

                return createBinarySparsePredictionMatrix(lilMatrix, numLabels, numNonZeroElements);
            }

            /**
             * @see `ISparsePredictor::predictSparse`
             */
            std::unique_ptr<BinarySparsePredictionMatrix> predictSparse(
                    const CsrConstView<const float32>& featureMatrix, uint32 numLabels) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                uint32 numFeatures = featureMatrix.getNumCols();
                BinaryLilMatrix lilMatrix(numExamples);
                uint32 numNonZeroElements = 0;
                const LabelVectorSet* labelVectorSetPtr = labelVectorSet_;

                if (labelVectorSetPtr && labelVectorSetPtr->getNumLabelVectors() > 0) {
                    const CsrConstView<const float32>* featureMatrixPtr = &featureMatrix;
                    BinaryLilMatrix* predictionMatrixPtr = &lilMatrix;
                    const Model* modelPtr = &model_;
                    const IProbabilityFunction* probabilityFunctionPtr = probabilityFunctionPtr_.get();

                    #pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) \
                    firstprivate(numFeatures) firstprivate(numLabels) firstprivate(modelPtr) \
                    firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) \
                    firstprivate(probabilityFunctionPtr) firstprivate(labelVectorSetPtr) schedule(dynamic) \
                    num_threads(numThreads_)
                    for (int64 i = 0; i < numExamples; i++) {
                        float64* scoreVector = new float64[numLabels] {};
                        applyRulesCsr(*modelPtr, numFeatures, featureMatrixPtr->row_indices_cbegin(i),
                                      featureMatrixPtr->row_indices_cend(i), featureMatrixPtr->row_values_cbegin(i),
                                      featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
                        // TODO
                        delete[] scoreVector;
                    }
                }

                return createBinarySparsePredictionMatrix(lilMatrix, numLabels, numNonZeroElements);
            }

    };

    /**
     * Allows to create instances of the type `IClassificationPredictor` that allow to predict whether individual labels
     * of given query examples are relevant or irrelevant by summing up the scores that are provided by the individual
     * rules of an existing rule-based model and transforming them into binary values according to the general F-measure
     * maximizer (GFM).
     */
    class GfmClassificationPredictorFactory final : public IClassificationPredictorFactory {

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
            GfmClassificationPredictorFactory(
                    std::unique_ptr<IProbabilityFunctionFactory> probabilityFunctionFactoryPtr, uint32 numThreads)
                : probabilityFunctionFactoryPtr_(std::move(probabilityFunctionFactoryPtr)), numThreads_(numThreads) {

            }

            /**
             * @see `IClassificationPredictorFactory::create`
             */
            std::unique_ptr<IClassificationPredictor> create(const RuleList& model,
                                                             const LabelVectorSet* labelVectorSet) const override {
                std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr = probabilityFunctionFactoryPtr_->create();
                return std::make_unique<GfmClassificationPredictor<RuleList>>(model, labelVectorSet,
                                                                              std::move(probabilityFunctionPtr),
                                                                              numThreads_);
            }

    };

    GfmClassificationPredictorConfig::GfmClassificationPredictorConfig(
            const std::unique_ptr<ILossConfig>& lossConfigPtr,
            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : lossConfigPtr_(std::move(lossConfigPtr)), multiThreadingConfigPtr_(std::move(multiThreadingConfigPtr)) {

    }

    std::unique_ptr<IClassificationPredictorFactory> GfmClassificationPredictorConfig::createClassificationPredictorFactory(
            const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
        std::unique_ptr<IProbabilityFunctionFactory> probabilityFunctionFactoryPtr =
            lossConfigPtr_->createProbabilityFunctionFactory();

        if (probabilityFunctionFactoryPtr) {
            uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
            return std::make_unique<GfmClassificationPredictorFactory>(std::move(probabilityFunctionFactoryPtr),
                                                                       numThreads);
        } else {
            // FIXME This will cause problems
            return nullptr;
        }
    }

    bool GfmClassificationPredictorConfig::isLabelVectorSetNeeded() const {
        return true;
    }

}
