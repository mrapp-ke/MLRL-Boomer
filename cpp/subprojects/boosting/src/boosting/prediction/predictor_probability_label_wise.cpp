#include "boosting/prediction/predictor_probability_label_wise.hpp"

#include "boosting/prediction/predictor_probability_common.hpp"
#include "boosting/prediction/transformation_probability_label_wise.hpp"

namespace boosting {

    /**
     * Allows to create instances of the type `IProbabilityPredictor` that allow to predict label-wise probabilities
     * for given query examples, which estimate the chance of individual labels to be relevant, by summing up the scores
     * that are provided by individual rules of an existing rule-based model and transforming the aggregated scores into
     * probabilities in [0, 1] according to a certain transformation function that is applied to each label
     * individually.
     */
    class LabelWiseProbabilityPredictorFactory final : public IProbabilityPredictorFactory {
        private:

            const std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactoryPtr_;

            const uint32 numThreads_;

        public:

            /**
             * @param marginalProbabilityFunctionFactoryPtr An unique pointer to an object of type
             *                                              `IMarginalProbabilityFunctionFactory` that allows to create
             *                                              implementations of the transformation function to be used to
             *                                              transform regression scores that are predicted for
             *                                              individual labels into probabilities
             * @param numThreads                            The number of CPU threads to be used to make predictions for
             *                                              different query examples in parallel. Must be at least 1
             */
            LabelWiseProbabilityPredictorFactory(
              std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactoryPtr,
              uint32 numThreads)
                : marginalProbabilityFunctionFactoryPtr_(std::move(marginalProbabilityFunctionFactoryPtr)),
                  numThreads_(numThreads) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IProbabilityPredictor> create(
              const CContiguousConstView<const float32>& featureMatrix, const RuleList& model,
              const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              uint32 numLabels) const override {
                std::unique_ptr<IProbabilityTransformation> probabilityTransformationPtr =
                  std::make_unique<LabelWiseProbabilityTransformation>(
                    marginalProbabilityFunctionFactoryPtr_->create(marginalProbabilityCalibrationModel));
                return std::make_unique<ProbabilityPredictor<CContiguousConstView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, numThreads_, std::move(probabilityTransformationPtr));
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IProbabilityPredictor> create(
              const CsrConstView<const float32>& featureMatrix, const RuleList& model,
              const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              uint32 numLabels) const override {
                std::unique_ptr<IProbabilityTransformation> probabilityTransformationPtr =
                  std::make_unique<LabelWiseProbabilityTransformation>(
                    marginalProbabilityFunctionFactoryPtr_->create(marginalProbabilityCalibrationModel));
                return std::make_unique<ProbabilityPredictor<CsrConstView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, numThreads_, std::move(probabilityTransformationPtr));
            }
    };

    LabelWiseProbabilityPredictorConfig::LabelWiseProbabilityPredictorConfig(
      const std::unique_ptr<ILossConfig>& lossConfigPtr,
      const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : lossConfigPtr_(lossConfigPtr), multiThreadingConfigPtr_(multiThreadingConfigPtr) {}

    std::unique_ptr<IProbabilityPredictorFactory> LabelWiseProbabilityPredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactoryPtr =
          lossConfigPtr_->createMarginalProbabilityFunctionFactory();

        if (marginalProbabilityFunctionFactoryPtr) {
            uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
            return std::make_unique<LabelWiseProbabilityPredictorFactory>(
              std::move(marginalProbabilityFunctionFactoryPtr), numThreads);
        } else {
            return nullptr;
        }
    }

    bool LabelWiseProbabilityPredictorConfig::isLabelVectorSetNeeded() const {
        return false;
    }

}
