#include "mlrl/boosting/prediction/predictor_score_output_wise.hpp"

#include "mlrl/boosting/prediction/predictor_score_common.hpp"

namespace boosting {

    /**
     * Allows to create instances of the type `IScorePredictor` that predict output-wise scores for given query examples
     * by summing up the scores that are provided by individual rules for each output individually.
     */
    class OutputWiseScorePredictorFactory final : public IScorePredictorFactory {
        private:

            const uint32 numThreads_;

        public:

            /**
             * @param numThreads The number of CPU threads to be used to make predictions for different query examples
             *                   in parallel. Must be at least 1
             */
            OutputWiseScorePredictorFactory(uint32 numThreads) : numThreads_(numThreads) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IScorePredictor> create(const CContiguousView<const float32>& featureMatrix,
                                                    const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                    uint32 numOutputs) const override {
                return std::make_unique<ScorePredictor<CContiguousView<const float32>, RuleList>>(
                  featureMatrix, model, numOutputs, numThreads_);
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IScorePredictor> create(const CsrView<const float32>& featureMatrix, const RuleList& model,
                                                    const LabelVectorSet* labelVectorSet,
                                                    uint32 numOutputs) const override {
                return std::make_unique<ScorePredictor<CsrView<const float32>, RuleList>>(featureMatrix, model,
                                                                                          numOutputs, numThreads_);
            }
    };

    OutputWiseScorePredictorConfig::OutputWiseScorePredictorConfig(
      GetterFunction<IMultiThreadingConfig> multiThreadingConfigGetter)
        : multiThreadingConfigGetter_(multiThreadingConfigGetter) {}

    std::unique_ptr<IScorePredictorFactory> OutputWiseScorePredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numOutputs) const {
        uint32 numThreads = multiThreadingConfigGetter_().getNumThreads(featureMatrix, numOutputs);
        return std::make_unique<OutputWiseScorePredictorFactory>(numThreads);
    }

    bool OutputWiseScorePredictorConfig::isLabelVectorSetNeeded() const {
        return false;
    }

}
