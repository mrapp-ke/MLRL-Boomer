#include "mlrl/boosting/prediction/predictor_score_output_wise.hpp"

#include "mlrl/boosting/prediction/predictor_score_common.hpp"

namespace boosting {

    /**
     * Allows to create instances of the type `IScorePredictor` that predict output-wise scores for given query examples
     * by summing up the scores that are provided by individual rules for each output individually.
     */
    class OutputWiseScorePredictorFactory final : public IScorePredictorFactory {
        private:

            const MultiThreadingSettings multiThreadingSettings_;

        public:

            /**
             * @param multiThreadingSettings An object of type `MultiThreadingSettings` that stores the settnigs to be
             *                               used for making predictions for different query examples in parallel
             */
            OutputWiseScorePredictorFactory(MultiThreadingSettings multiThreadingSettings)
                : multiThreadingSettings_(multiThreadingSettings) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IScorePredictor> create(const CContiguousView<const float32>& featureMatrix,
                                                    const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                    uint32 numOutputs) const override {
                return std::make_unique<ScorePredictor<CContiguousView<const float32>, RuleList>>(
                  featureMatrix, model, numOutputs, multiThreadingSettings_);
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IScorePredictor> create(const CsrView<const float32>& featureMatrix, const RuleList& model,
                                                    const LabelVectorSet* labelVectorSet,
                                                    uint32 numOutputs) const override {
                return std::make_unique<ScorePredictor<CsrView<const float32>, RuleList>>(
                  featureMatrix, model, numOutputs, multiThreadingSettings_);
            }
    };

    OutputWiseScorePredictorConfig::OutputWiseScorePredictorConfig(
      ReadableProperty<IMultiThreadingConfig> multiThreadingConfig)
        : multiThreadingConfig_(multiThreadingConfig) {}

    std::unique_ptr<IScorePredictorFactory> OutputWiseScorePredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numOutputs) const {
        MultiThreadingSettings multiThreadingSettings =
          multiThreadingConfig_.get().getSettings(featureMatrix, numOutputs);
        return std::make_unique<OutputWiseScorePredictorFactory>(multiThreadingSettings);
    }

    bool OutputWiseScorePredictorConfig::isLabelVectorSetNeeded() const {
        return false;
    }

}
