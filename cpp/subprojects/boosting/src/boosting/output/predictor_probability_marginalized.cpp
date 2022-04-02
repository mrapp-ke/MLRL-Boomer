#include "boosting/output/predictor_probability_marginalized.hpp"


namespace boosting {

    /**
     * An implementation of the type `IProbabilityPredictor` that allows to predict marginalized probabilities for given
     * for given query examples, which estimate the chance of individual labels to be relevant, by summing up the scores
     * that are provided by individual rules of an existing rule-based models and comparing the aggregated score vector
     * to the known label vectors according to a certain distance measure. The probability for an individual label
     * calculates as the sum of the distances that have been obtained all label vectors, where the respective label is
     * specified to be relevant, divided by the total sum of all distances.
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
                // TODO
                return nullptr;
            }

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<float64>> predict(const CsrConstView<const float32>& featureMatrix,
                                                                    uint32 numLabels) const override {
                // TODO
                return nullptr;
            }

    };

    /**
     * Allows to create instances of the type `IProbabilityPredictor` that allow to predict marginalized probabilities
     * for given query examples, which estimate the chance of individual labels to be relevant, by summing up the scores
     * that are provided by individual rules of an existing rule-based model and comparing the aggregated score vector
     * to the known label vectors according to a certain distance measure. The probability for an individual label
     * calculates as the sum of the distances that have been obtained all label vectors, where the respective label is
     * specified to be relevant, divided by the total sum of all distances.
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

}
