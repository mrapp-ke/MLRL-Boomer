#include "boosting/prediction/predictor_probability_marginalized.hpp"

#include "boosting/prediction/predictor_probability_common.hpp"
#include "boosting/prediction/transformation_probability_marginalized.hpp"

#include <stdexcept>

namespace boosting {

    template<typename FeatureMatrix, typename Model>
    static inline std::unique_ptr<IProbabilityPredictor> createPredictor(
      const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels, uint32 numThreads,
      const LabelVectorSet* labelVectorSet, const IProbabilityFunctionFactory& probabilityFunctionFactory) {
        if (!labelVectorSet) {
            throw std::runtime_error(
              "Information about the label vectors that have been encountered in the training data is required for "
              "predicting binary labels, but no such information is provided by the model. Most probably, the model "
              "was intended to use a different prediction method when it has been trained.");
        }

        std::unique_ptr<IProbabilityTransformation> probabilityTransformationPtr;

        if (labelVectorSet->getNumLabelVectors() > 0) {
            probabilityTransformationPtr = std::make_unique<MarginalizedProbabilityTransformation>(
              *labelVectorSet, probabilityFunctionFactory.create());
        }

        return std::make_unique<ProbabilityPredictor<FeatureMatrix, Model>>(featureMatrix, model, numLabels, numThreads,
                                                                            std::move(probabilityTransformationPtr));
    }

    /**
     * Allows to create instances of the type `IProbabilityPredictor` that allow to predict marginalized probabilities
     * for given query examples, which estimate the chance of individual labels to be relevant, by summing up the scores
     * that are provided by individual rules of an existing rule-based model and comparing the aggregated score vector
     * to the known label vectors according to a certain distance measure. The probability for an individual label
     * calculates as the sum of the distances that have been obtained for all label vectors, where the respective label
     * is specified to be relevant, divided by the total sum of all distances.
     */
    class MarginalizedProbabilityPredictorFactory final : public IProbabilityPredictorFactory {
        private:

            std::unique_ptr<IProbabilityFunctionFactory> probabilityFunctionFactoryPtr_;

            const uint32 numThreads_;

        public:

            /**
             * @param probabilityFunctionFactoryPtr An unique pointer to an object of type `IProbabilityFunctionFactory`
             *                                      that allows to create implementations of the transformation function
             *                                      to be used to transform predicted scores into probabilities
             * @param numThreads                    The number of CPU threads to be used to make predictions for
             *                                      different query examples in parallel. Must be at least 1
             */
            MarginalizedProbabilityPredictorFactory(
              std::unique_ptr<IProbabilityFunctionFactory> probabilityFunctionFactoryPtr, uint32 numThreads)
                : probabilityFunctionFactoryPtr_(std::move(probabilityFunctionFactoryPtr)), numThreads_(numThreads) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IProbabilityPredictor> create(const CContiguousConstView<const float32>& featureMatrix,
                                                          const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                          uint32 numLabels) const override {
                return createPredictor(featureMatrix, model, numLabels, numThreads_, labelVectorSet,
                                       *probabilityFunctionFactoryPtr_);
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IProbabilityPredictor> create(const CsrConstView<const float32>& featureMatrix,
                                                          const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                          uint32 numLabels) const override {
                return createPredictor(featureMatrix, model, numLabels, numThreads_, labelVectorSet,
                                       *probabilityFunctionFactoryPtr_);
            }
    };

    MarginalizedProbabilityPredictorConfig::MarginalizedProbabilityPredictorConfig(
      const std::unique_ptr<ILossConfig>& lossConfigPtr,
      const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : lossConfigPtr_(std::move(lossConfigPtr)), multiThreadingConfigPtr_(std::move(multiThreadingConfigPtr)) {}

    std::unique_ptr<IProbabilityPredictorFactory> MarginalizedProbabilityPredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        std::unique_ptr<IProbabilityFunctionFactory> probabilityFunctionFactoryPtr =
          lossConfigPtr_->createProbabilityFunctionFactory();

        if (probabilityFunctionFactoryPtr) {
            uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
            return std::make_unique<MarginalizedProbabilityPredictorFactory>(std::move(probabilityFunctionFactoryPtr),
                                                                             numThreads);
        } else {
            return nullptr;
        }
    }

    bool MarginalizedProbabilityPredictorConfig::isLabelVectorSetNeeded() const {
        return true;
    }

}
