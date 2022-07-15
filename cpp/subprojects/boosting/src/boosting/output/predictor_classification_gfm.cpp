#include "boosting/output/predictor_classification_gfm.hpp"
#include "common/data/vector_sparse_array.hpp"
#include "common/math/math.hpp"
#include "predictor_common.hpp"
#include "predictor_probability_common.hpp"
#include "omp.h"
#include <algorithm>


namespace boosting {

    static inline float64 calculateMarginalizedProbabilities(float64* probabilities, uint32 numLabels,
                                                             const float64* jointProbabilities,
                                                             float64 sumOfJointProbabilities,
                                                             const LabelVectorSet& labelVectorSet) {
        float64 nullVectorProbability = 0;
        uint32 i = 0;

        for (auto it = labelVectorSet.cbegin(); it != labelVectorSet.cend(); it++) {
            const auto& entry = *it;
            const std::unique_ptr<LabelVector>& labelVectorPtr = entry.first;
            uint32 numRelevantLabels = labelVectorPtr->getNumElements();
            float64 normalizedJointProbability = divideOrZero(jointProbabilities[i], sumOfJointProbabilities);

            if (numRelevantLabels > 0) {
                LabelVector::const_iterator labelIndexIterator = labelVectorPtr->cbegin();

                for (uint32 j = 0; j < numRelevantLabels; j++) {
                    uint32 labelIndex = labelIndexIterator[j];
                    probabilities[(numLabels * labelIndex) + numRelevantLabels] += normalizedJointProbability;
                }
            } else {
                nullVectorProbability = normalizedJointProbability;
            }

            i++;
        }

        return nullVectorProbability;
    }

    static inline float64 createAndEvaluateLabelVector(SparseArrayVector<float64>::iterator iterator, uint32 numLabels,
                                                       const float64* probabilities, uint32 k) {
        for (uint32 i = 0; i < numLabels; i++) {
            uint32 offset = i * numLabels;
            float64 weightedProbability = 0;

            for (uint32 j = 0; j < numLabels; j++) {
                float64 probability = probabilities[offset + j];
                weightedProbability += (2 * probability) / (float64) (j + k + 1);
            }

            IndexedValue<float64>& entry = iterator[i];
            entry.index = i;
            entry.value = weightedProbability;
        }

        std::partial_sort(iterator, &iterator[k], &iterator[numLabels],
                          [=](const IndexedValue<float64>& a, const IndexedValue<float64>& b) {
            return a.value > b.value;
        });

        float64 quality = 0;

        for (uint32 i = 0; i < k; i++) {
            quality += iterator[i].value;
        }

        return quality;
    }

    static inline uint32 storePrediction(const SparseArrayVector<float64>& tmpVector,
                                       DensePredictionMatrix<uint8>::value_iterator predictionIterator) {
        uint32 numRelevantLabels = tmpVector.getNumElements();
        SparseArrayVector<float64>::const_iterator iterator = tmpVector.cbegin();

        for (uint32 i = 0; i < numRelevantLabels; i++) {
            uint32 labelIndex = iterator[i].index;
            predictionIterator[labelIndex] = 1;
        }

        return numRelevantLabels;
    }

    static inline uint32 storePrediction(SparseArrayVector<float64>& tmpVector, BinaryLilMatrix::row row) {
        uint32 numRelevantLabels = tmpVector.getNumElements();

        if (numRelevantLabels > 0) {
            SparseArrayVector<float64>::iterator iterator = tmpVector.begin();
            std::sort(iterator, tmpVector.end(), [=](const IndexedValue<float64>& a, const IndexedValue<float64>& b) {
                return a.index < b.index;
            });

            for (uint32 i = 0; i < numRelevantLabels; i++) {
                row.emplace_back(iterator[i].index);
            }
        }

        return numRelevantLabels;
    }

    template<typename Prediction>
    static inline uint32 predictGfm(const float64* scoresBegin, const float64* scoresEnd, Prediction prediction,
                                    uint32 numLabels, const IProbabilityFunction& probabilityFunction,
                                    const LabelVectorSet& labelVectorSet, uint32 numLabelVectors) {
        float64* jointProbabilities = new float64[numLabelVectors];
        float64 sumOfJointProbabilities = calculateJointProbabilities(scoresBegin, scoresEnd, jointProbabilities,
                                                                      probabilityFunction, labelVectorSet);
        float64* marginalProbabilities = new float64[numLabels * numLabels] {}; // TODO Use sparse representation
        float64 bestQuality = calculateMarginalizedProbabilities(marginalProbabilities, numLabels, jointProbabilities,
                                                                 sumOfJointProbabilities, labelVectorSet);
        delete[] jointProbabilities;

        SparseArrayVector<float64> tmpVector(numLabels);
        tmpVector.setNumElements(0, false);
        SparseArrayVector<float64> tmpVector2(numLabels);
        SparseArrayVector<float64>* bestVectorPtr = &tmpVector;
        SparseArrayVector<float64>* tmpVectorPtr = &tmpVector2;

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 k = i + 1;
            float64 quality = createAndEvaluateLabelVector(tmpVectorPtr->begin(), numLabels, marginalProbabilities, k);

            if (quality > bestQuality) {
                bestQuality = quality;
                tmpVectorPtr->setNumElements(k, false);
                SparseArrayVector<float64>* tmpPtr = bestVectorPtr;
                bestVectorPtr = tmpVectorPtr;
                tmpVectorPtr = tmpPtr;
            }
        }

        delete[] marginalProbabilities;
        return storePrediction(*bestVectorPtr, prediction);
    }

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

                if (labelVectorSetPtr) {
                    uint32 numLabelVectors = labelVectorSetPtr->getNumLabelVectors();

                    if (numLabelVectors > 0) {
                        const CContiguousConstView<const float32>* featureMatrixPtr = &featureMatrix;
                        CContiguousView<uint8>* predictionMatrixRawPtr = predictionMatrixPtr.get();
                        const Model* modelPtr = &model_;
                        const IProbabilityFunction* probabilityFunctionPtr = probabilityFunctionPtr_.get();

                        #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) \
                        firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixRawPtr) \
                        firstprivate(probabilityFunctionPtr) firstprivate(labelVectorSetPtr) \
                        firstprivate(numLabelVectors) schedule(dynamic) num_threads(numThreads_)
                        for (int64 i = 0; i < numExamples; i++) {
                            float64* scoreVector = new float64[numLabels] {};
                            applyRules(*modelPtr, featureMatrixPtr->row_values_cbegin(i),
                                       featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
                            predictGfm(scoreVector, &scoreVector[numLabels],
                                       predictionMatrixRawPtr->row_values_begin(i), numLabels, *probabilityFunctionPtr,
                                       *labelVectorSetPtr, numLabelVectors);
                            delete[] scoreVector;
                        }
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

                if (labelVectorSetPtr) {
                    uint32 numLabelVectors = labelVectorSetPtr->getNumLabelVectors();

                    if (numLabelVectors > 0) {
                        const CsrConstView<const float32>* featureMatrixPtr = &featureMatrix;
                        CContiguousView<uint8>* predictionMatrixRawPtr = predictionMatrixPtr.get();
                        const Model* modelPtr = &model_;
                        const IProbabilityFunction* probabilityFunctionPtr = probabilityFunctionPtr_.get();

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
                            predictGfm(scoreVector, &scoreVector[numLabels],
                                       predictionMatrixRawPtr->row_values_begin(i), numLabels, *probabilityFunctionPtr,
                                       *labelVectorSetPtr, numLabelVectors);
                            delete[] scoreVector;
                        }
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

                if (labelVectorSetPtr) {
                    uint32 numLabelVectors = labelVectorSetPtr->getNumLabelVectors();

                    if (numLabelVectors > 0) {
                        const CContiguousConstView<const float32>* featureMatrixPtr = &featureMatrix;
                        BinaryLilMatrix* predictionMatrixPtr = &lilMatrix;
                        const Model* modelPtr = &model_;
                        const IProbabilityFunction* probabilityFunctionPtr = probabilityFunctionPtr_.get();

                        #pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) \
                        firstprivate(numLabels) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
                        firstprivate(predictionMatrixPtr) firstprivate(probabilityFunctionPtr) \
                        firstprivate(labelVectorSetPtr) firstprivate(numLabelVectors) schedule(dynamic) \
                        num_threads(numThreads_)
                        for (int64 i = 0; i < numExamples; i++) {
                            float64* scoreVector = new float64[numLabels] {};
                            applyRules(*modelPtr, featureMatrixPtr->row_values_cbegin(i),
                                       featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
                            numNonZeroElements += predictGfm<BinaryLilMatrix::row>(scoreVector, &scoreVector[numLabels],
                                                                                   (*predictionMatrixPtr)[i], numLabels,
                                                                                   *probabilityFunctionPtr,
                                                                                   *labelVectorSetPtr, numLabelVectors);
                            delete[] scoreVector;
                        }
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

                if (labelVectorSetPtr) {
                    uint32 numLabelVectors = labelVectorSetPtr->getNumLabelVectors();

                    if (numLabelVectors > 0) {
                        const CsrConstView<const float32>* featureMatrixPtr = &featureMatrix;
                        BinaryLilMatrix* predictionMatrixPtr = &lilMatrix;
                        const Model* modelPtr = &model_;
                        const IProbabilityFunction* probabilityFunctionPtr = probabilityFunctionPtr_.get();

                        #pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) \
                        firstprivate(numFeatures) firstprivate(numLabels) firstprivate(modelPtr) \
                        firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) \
                        firstprivate(probabilityFunctionPtr) firstprivate(labelVectorSetPtr) \
                        firstprivate(numLabelVectors) schedule(dynamic) num_threads(numThreads_)
                        for (int64 i = 0; i < numExamples; i++) {
                            float64* scoreVector = new float64[numLabels] {};
                            applyRulesCsr(*modelPtr, numFeatures, featureMatrixPtr->row_indices_cbegin(i),
                                          featureMatrixPtr->row_indices_cend(i), featureMatrixPtr->row_values_cbegin(i),
                                          featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
                            numNonZeroElements += predictGfm<BinaryLilMatrix::row>(scoreVector, &scoreVector[numLabels],
                                                                                   (*predictionMatrixPtr)[i], numLabels,
                                                                                   *probabilityFunctionPtr,
                                                                                   *labelVectorSetPtr, numLabelVectors);
                            delete[] scoreVector;
                        }
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
            return nullptr;
        }
    }

    bool GfmClassificationPredictorConfig::isLabelVectorSetNeeded() const {
        return true;
    }

}
