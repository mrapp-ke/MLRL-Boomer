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

            CContiguousView<float64>& predictionMatrix_;

            const IProbabilityTransformation& probabilityTransformation_;

        public:

            /**
             * @param scoreMatrix               A reference to an object of type `CContiguousView` that stores the
             *                                  aggregated scores
             * @param predictionMatrix          A reference to an object of type `CContiguousView` that stores the
             *                                  probability estimates
             * @param probabilityTransformation A reference to an object of type `IProbabilityTransformation` that
             *                                  should be used to transform aggregated scores into probability estimates
             */
            ProbabilityPredictionDelegate(CContiguousView<float64>& scoreMatrix,
                                          CContiguousView<float64>& predictionMatrix,
                                          const IProbabilityTransformation& probabilityTransformation)
                : scoreMatrix_(scoreMatrix), predictionMatrix_(predictionMatrix),
                  probabilityTransformation_(probabilityTransformation) {}

            /**
             * @see `PredictionDispatcher::IPredictionDelegate::predictForExample`
             */
            void predictForExample(const FeatureMatrix& featureMatrix, typename Model::const_iterator rulesBegin,
                                   typename Model::const_iterator rulesEnd, uint32 threadIndex, uint32 exampleIndex,
                                   uint32 predictionIndex) const override {
                ScorePredictionDelegate<FeatureMatrix, Model>(scoreMatrix_)
                  .predictForExample(featureMatrix, rulesBegin, rulesEnd, threadIndex, exampleIndex, predictionIndex);
                probabilityTransformation_.apply(scoreMatrix_.row_values_cbegin(predictionIndex),
                                                 scoreMatrix_.row_values_cend(predictionIndex),
                                                 predictionMatrix_.row_values_begin(predictionIndex),
                                                 predictionMatrix_.row_values_end(predictionIndex));
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

            class IncrementalPredictor final : public IIncrementalPredictor<DensePredictionMatrix<float64>> {
                private:

                    const FeatureMatrix& featureMatrix_;

                    uint32 numThreads_;

                    std::shared_ptr<IProbabilityTransformation> probabilityTransformationPtr_;

                    DensePredictionMatrix<float64> scoreMatrix_;

                    DensePredictionMatrix<float64> predictionMatrix_;

                    typename Model::const_iterator current_;

                    typename Model::const_iterator end_;

                public:

                    IncrementalPredictor(const ProbabilityPredictor& predictor, uint32 minRules, uint32 maxRules,
                                         std::shared_ptr<IProbabilityTransformation> probabilityTransformationPtr)
                        : featureMatrix_(predictor.featureMatrix_), numThreads_(predictor.numThreads_),
                          probabilityTransformationPtr_(probabilityTransformationPtr),
                          scoreMatrix_(
                            DensePredictionMatrix<float64>(featureMatrix_.getNumRows(), predictor.numLabels_, true)),
                          predictionMatrix_(
                            DensePredictionMatrix<float64>(featureMatrix_.getNumRows(), predictor.numLabels_)),
                          current_(predictor.model_.used_cbegin(maxRules) + (minRules - 1)),
                          end_(predictor.model_.used_cend(maxRules)) {}

                    uint32 getNumNext() const override {
                        return (uint32) (end_ - current_);
                    }

                    DensePredictionMatrix<float64>& applyNext(uint32 stepSize) override {
                        typename Model::const_iterator next = current_ + std::min(stepSize, this->getNumNext());
                        ProbabilityPredictionDelegate<FeatureMatrix, Model> delegate(scoreMatrix_, predictionMatrix_,
                                                                                     *probabilityTransformationPtr_);
                        PredictionDispatcher<float64, FeatureMatrix, Model>().predict(delegate, featureMatrix_,
                                                                                      current_, next, numThreads_);
                        current_ = next;
                        return predictionMatrix_;
                    }
            };

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            uint32 numLabels_;

            uint32 numThreads_;

            std::shared_ptr<IProbabilityTransformation> probabilityTransformationPtr_;

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
                    ProbabilityPredictionDelegate<FeatureMatrix, Model> delegate(
                      *predictionMatrixPtr, *predictionMatrixPtr, *probabilityTransformationPtr_);
                    PredictionDispatcher<float64, FeatureMatrix, Model>().predict(
                      delegate, featureMatrix_, model_.used_cbegin(maxRules), model_.used_cend(maxRules), numThreads_);
                }

                return predictionMatrixPtr;
            }

            /**
             * @see `IPredictor::canPredictIncrementally`
             */
            bool canPredictIncrementally() const override {
                return true;
            }

            /**
             * @see `IPredictor::createIncrementalPredictor`
             */
            std::unique_ptr<IIncrementalPredictor<DensePredictionMatrix<float64>>> createIncrementalPredictor(
              uint32 minRules, uint32 maxRules) const override {
                assertGreaterOrEqual<uint32>("minRules", minRules, 1);
                if (maxRules != 0) assertGreater<uint32>("maxRules", maxRules, minRules);
                return std::make_unique<IncrementalPredictor>(*this, minRules, maxRules, probabilityTransformationPtr_);
            }
    };

}
