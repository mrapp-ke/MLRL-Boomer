/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/prediction/predictor_score_common.hpp"
#include "boosting/prediction/transformation_probability.hpp"
#include "common/prediction/predictor_probability.hpp"

namespace boosting {

    /**
     * An implementation of the type `PredictionDispatcher::IPredictionDelegate` that transforms the aggregated scores
     * that are predicted by a rule-based model into probability estimates.
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples.
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class ProbabilityPredictionDelegate final
        : public PredictionDispatcher<float64, FeatureMatrix, Model>::IPredictionDelegate {
        private:

            CContiguousView<float64>& scoreMatrix_;

            const IProbabilityTransformation& probabilityTransformation_;

        public:

            /**
             * @param scoreMatrix               A reference to an object of type `CContiguousView` that stores the
             *                                  aggregated scores
             * @param probabilityTransformation A reference to an object of type `IProbabilityTransformation` that
             *                                  should be used to transform aggregated scores into probability estimates
             */
            ProbabilityPredictionDelegate(CContiguousView<float64>& scoreMatrix,
                                          const IProbabilityTransformation& probabilityTransformation)
                : scoreMatrix_(scoreMatrix), probabilityTransformation_(probabilityTransformation) {}

            /**
             * @see `PredictionDispatcher::IPredictionDelegate::predictForExample`
             */
            void predictForExample(const FeatureMatrix& featureMatrix, const Model& model, uint32 maxRules,
                                   uint32 threadIndex, uint32 exampleIndex, uint32 predictionIndex) const override {
                ScorePredictionDelegate<FeatureMatrix, Model>(scoreMatrix_)
                  .predictForExample(featureMatrix, model, maxRules, threadIndex, exampleIndex, predictionIndex);
                probabilityTransformation_.apply(scoreMatrix_.row_values_begin(predictionIndex),
                                                 scoreMatrix_.row_values_end(predictionIndex));
            }
    };

    /**
     * An implementation of the type `IProbabilityPredictor` that allows to predict label-wise probability estimates for
     * given query examples, estimating the chance of individual labels to be relevant, by summing up the scores that
     * are predicted by individual rules in a rule-based model and transforming the aggregated scores into probabilities
     * in [0, 1] according to an `IProbabilityTransformation`.
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class ProbabilityPredictor final : public IProbabilityPredictor {
        private:

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            uint32 numLabels_;

            uint32 numThreads_;

            std::unique_ptr<IProbabilityTransformation> probabilityTransformationPtr_;

        public:

            /**
             * @param featureMatrix                 A reference to an object of template type `FeatureMatrix` that
             *                                      provides row-wise access to the feature values of the query examples
             * @param model                         A reference to an object of template type `Model` that should be
             *                                      used to obtain predictions
             * @param numLabels                     The number of labels to predict for
             * @param numThreads                    The number of CPU threads to be used to make predictions for
             *                                      different query examples in parallel. Must be at least 1
             * @param probabilityTransformationPtr  An unique pointer to an object of type `IProbabilityTransformation`
             *                                      that should be used to transform aggregated scores into probability
             *                                      estimates or a null pointer, if all probabilities should be set to
             *                                      zero
             */
            ProbabilityPredictor(const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels,
                                 uint32 numThreads,
                                 std::unique_ptr<IProbabilityTransformation> probabilityTransformationPtr)
                : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels), numThreads_(numThreads),
                  probabilityTransformationPtr_(std::move(probabilityTransformationPtr)) {}

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<float64>> predict(uint32 maxRules) const override {
                uint32 numExamples = featureMatrix_.getNumRows();
                std::unique_ptr<DensePredictionMatrix<float64>> predictionMatrixPtr =
                  std::make_unique<DensePredictionMatrix<float64>>(numExamples, numLabels_, true);

                if (probabilityTransformationPtr_) {
                    ProbabilityPredictionDelegate<FeatureMatrix, Model> delegate(*predictionMatrixPtr,
                                                                                 *probabilityTransformationPtr_);
                    PredictionDispatcher<float64, FeatureMatrix, Model>().predict(delegate, featureMatrix_, model_,
                                                                                  maxRules, numThreads_);
                }

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
              uint32 minRules, uint32 maxRules) const override {
                throw std::runtime_error(
                  "The rule learner does not support to predict probability estimates incrementally");
            }
    };

}
