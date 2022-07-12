#include "boosting/output/predictor_classification_gfm.hpp"


namespace boosting {

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
                // TODO
                return nullptr;
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
