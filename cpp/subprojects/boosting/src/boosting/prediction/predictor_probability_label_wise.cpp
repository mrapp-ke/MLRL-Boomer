#include "boosting/prediction/predictor_probability_label_wise.hpp"

#include "boosting/prediction/predictor_probability_common.hpp"
#include "boosting/prediction/transformation_probability_label_wise.hpp"

namespace boosting {

    /**
     * An implementation of the type `IProbabilityPredictor` that allows to predict label-wise probabilities for given
     * query examples, which estimate the chance of individual labels to be relevant, by summing up the scores that are
     * provided by individual rules of an existing rule-based model and transforming the aggregated scores into
     * probabilities in [0, 1] according to a certain transformation function that is applied to each label
     * individually.
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class LabelWiseProbabilityPredictor final : public IProbabilityPredictor {
        private:

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            uint32 numLabels_;

            uint32 numThreads_;

            std::unique_ptr<IProbabilityTransformation> probabilityTransformationPtr_;

        public:

            /**
             * @param featureMatrix             A reference to an object of template type `FeatureMatrix` that provides
             *                                  row-wise access to the feature values of the query examples
             * @param model                     A reference to an object of template type `Model` that should be used to
             *                                  obtain predictions
             * @param numLabels                 The number of labels to predict for
             * @param probabilityFunctionPtr    An unique pointer to an object of type `IProbabilityFunction` that
             *                                  should be used to transform predicted scores into probabilities
             * @param numThreads                The number of CPU threads to be used to make predictions for different
             *                                  query examples in parallel. Must be at least 1
             */
            LabelWiseProbabilityPredictor(const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels,
                                          std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr,
                                          uint32 numThreads)
                : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels), numThreads_(numThreads),
                  probabilityTransformationPtr_(
                    std::make_unique<LabelWiseProbabilityTransformation>(std::move(probabilityFunctionPtr))) {}

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<float64>> predict(uint32 maxRules) const override {
                uint32 numExamples = featureMatrix_.getNumRows();
                std::unique_ptr<DensePredictionMatrix<float64>> predictionMatrixPtr =
                  std::make_unique<DensePredictionMatrix<float64>>(numExamples, numLabels_, true);
                ProbabilityPredictionDelegate<FeatureMatrix, Model> delegate(*predictionMatrixPtr,
                                                                             *probabilityTransformationPtr_);
                PredictionDispatcher<float64, FeatureMatrix, Model>().predict(delegate, featureMatrix_, model_,
                                                                              maxRules, numThreads_);
                return predictionMatrixPtr;
            }

            /**
             * @see `IPredictor::canPredictIncrementally`
             */
            bool canPredictIncrementally() const override {
                return false;
            }

            /**
             * @see `IPredictor::createIncrementalPredictor`
             */
            std::unique_ptr<IIncrementalPredictor<DensePredictionMatrix<float64>>> createIncrementalPredictor(
              uint32 maxRules) const override {
                throw std::runtime_error(
                  "The rule learner does not support to predict probability estimates incrementally");
            }
    };

    /**
     * Allows to create instances of the type `IProbabilityPredictor` that allow to predict label-wise probabilities
     * for given query examples, which estimate the chance of individual labels to be relevant, by summing up the scores
     * that are provided by individual rules of an existing rule-based model and transforming the aggregated scores into
     * probabilities in [0, 1] according to a certain transformation function that is applied to each label
     * individually.
     */
    class LabelWiseProbabilityPredictorFactory final : public IProbabilityPredictorFactory {
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
            LabelWiseProbabilityPredictorFactory(
              std::unique_ptr<IProbabilityFunctionFactory> probabilityFunctionFactoryPtr, uint32 numThreads)
                : probabilityFunctionFactoryPtr_(std::move(probabilityFunctionFactoryPtr)), numThreads_(numThreads) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IProbabilityPredictor> create(const CContiguousConstView<const float32>& featureMatrix,
                                                          const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                          uint32 numLabels) const override {
                std::unique_ptr<IProbabilityTransformation> probabilityTransformationPtr =
                  std::make_unique<LabelWiseProbabilityTransformation>(probabilityFunctionFactoryPtr_->create());
                return std::make_unique<ProbabilityPredictor<CContiguousConstView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, numThreads_, std::move(probabilityTransformationPtr));
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IProbabilityPredictor> create(const CsrConstView<const float32>& featureMatrix,
                                                          const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                          uint32 numLabels) const override {
                std::unique_ptr<IProbabilityTransformation> probabilityTransformationPtr =
                  std::make_unique<LabelWiseProbabilityTransformation>(probabilityFunctionFactoryPtr_->create());
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
        std::unique_ptr<IProbabilityFunctionFactory> probabilityFunctionFactoryPtr =
          lossConfigPtr_->createProbabilityFunctionFactory();

        if (probabilityFunctionFactoryPtr) {
            uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
            return std::make_unique<LabelWiseProbabilityPredictorFactory>(std::move(probabilityFunctionFactoryPtr),
                                                                          numThreads);
        } else {
            return nullptr;
        }
    }

    bool LabelWiseProbabilityPredictorConfig::isLabelVectorSetNeeded() const {
        return false;
    }

}
