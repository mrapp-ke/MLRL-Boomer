/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/prediction/predictor_score_common.hpp"
#include "mlrl/boosting/prediction/transformation_probability.hpp"
#include "mlrl/common/prediction/predictor_probability.hpp"

#include <memory>
#include <utility>

namespace boosting {

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

            class PredictionDelegate final
                : public PredictionDispatcher<float64, FeatureMatrix, Model>::IPredictionDelegate {
                private:

                    CContiguousView<float64>& scoreMatrix_;

                    CContiguousView<float64>& predictionMatrix_;

                    const IProbabilityTransformation& probabilityTransformation_;

                public:

                    PredictionDelegate(CContiguousView<float64>& scoreMatrix,
                                       CContiguousView<float64>& predictionMatrix,
                                       const IProbabilityTransformation& probabilityTransformation)
                        : scoreMatrix_(scoreMatrix), predictionMatrix_(predictionMatrix),
                          probabilityTransformation_(probabilityTransformation) {}

                    void predictForExample(const FeatureMatrix& featureMatrix,
                                           typename Model::const_iterator rulesBegin,
                                           typename Model::const_iterator rulesEnd, uint32 threadIndex,
                                           uint32 exampleIndex, uint32 predictionIndex) const override {
                        ScorePredictionDelegate<FeatureMatrix, Model>(scoreMatrix_)
                          .predictForExample(featureMatrix, rulesBegin, rulesEnd, threadIndex, exampleIndex,
                                             predictionIndex);
                        probabilityTransformation_.apply(scoreMatrix_.values_cbegin(predictionIndex),
                                                         scoreMatrix_.values_cend(predictionIndex),
                                                         predictionMatrix_.values_begin(predictionIndex),
                                                         predictionMatrix_.values_end(predictionIndex));
                    }
            };

            class IncrementalPredictor final
                : public AbstractIncrementalPredictor<FeatureMatrix, Model, DensePredictionMatrix<float64>> {
                private:

                    const std::shared_ptr<IProbabilityTransformation> probabilityTransformationPtr_;

                    DensePredictionMatrix<float64> scoreMatrix_;

                    DensePredictionMatrix<float64> predictionMatrix_;

                protected:

                    DensePredictionMatrix<float64>& applyNext(const FeatureMatrix& featureMatrix,
                                                              MultiThreadingSettings multiThreadingSettings,
                                                              typename Model::const_iterator rulesBegin,
                                                              typename Model::const_iterator rulesEnd) override {
                        if (probabilityTransformationPtr_) {
                            PredictionDelegate delegate(scoreMatrix_.getView(), predictionMatrix_.getView(),
                                                        *probabilityTransformationPtr_);
                            PredictionDispatcher<float64, FeatureMatrix, Model>().predict(
                              delegate, featureMatrix, rulesBegin, rulesEnd, multiThreadingSettings);
                        }

                        return predictionMatrix_;
                    }

                public:

                    IncrementalPredictor(const ProbabilityPredictor& predictor, uint32 maxRules,
                                         std::shared_ptr<IProbabilityTransformation> probabilityTransformationPtr)
                        : AbstractIncrementalPredictor<FeatureMatrix, Model, DensePredictionMatrix<float64>>(
                            predictor.featureMatrix_, predictor.model_, predictor.multiThreadingSettings_, maxRules),
                          probabilityTransformationPtr_(probabilityTransformationPtr),
                          scoreMatrix_(predictor.featureMatrix_.numRows, predictor.numLabels_,
                                       probabilityTransformationPtr_ != nullptr),
                          predictionMatrix_(predictor.featureMatrix_.numRows, predictor.numLabels_,
                                            probabilityTransformationPtr_ == nullptr) {}
            };

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            const uint32 numLabels_;

            const MultiThreadingSettings multiThreadingSettings_;

            const std::shared_ptr<IProbabilityTransformation> probabilityTransformationPtr_;

        public:

            /**
             * @param featureMatrix                 A reference to an object of template type `FeatureMatrix` that
             *                                      provides row-wise access to the feature values of the query examples
             * @param model                         A reference to an object of template type `Model` that should be
             *                                      used to obtain predictions
             * @param numLabels                     The number of labels to predict for
             * @param multiThreadingSettings        An object of type `MultiThreadingSettings` that stores the settings
             *                                      to be used for making predictions for different query examples in
             *                                      parallel
             * @param probabilityTransformationPtr  An unique pointer to an object of type `IProbabilityTransformation`
             *                                      that should be used to transform aggregated scores into probability
             *                                      estimates or a null pointer, if all probabilities should be set to
             *                                      zero
             */
            ProbabilityPredictor(const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels,
                                 MultiThreadingSettings multiThreadingSettings,
                                 std::unique_ptr<IProbabilityTransformation> probabilityTransformationPtr)
                : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels),
                  multiThreadingSettings_(multiThreadingSettings),
                  probabilityTransformationPtr_(std::move(probabilityTransformationPtr)) {}

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<float64>> predict(uint32 maxRules) const override {
                std::unique_ptr<DensePredictionMatrix<float64>> predictionMatrixPtr =
                  std::make_unique<DensePredictionMatrix<float64>>(featureMatrix_.numRows, numLabels_, true);

                if (probabilityTransformationPtr_) {
                    PredictionDelegate delegate(predictionMatrixPtr->getView(), predictionMatrixPtr->getView(),
                                                *probabilityTransformationPtr_);
                    PredictionDispatcher<float64, FeatureMatrix, Model>().predict(
                      delegate, featureMatrix_, model_.used_cbegin(maxRules), model_.used_cend(maxRules),
                      multiThreadingSettings_);
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
              uint32 maxRules) const override {
                if (maxRules != 0) util::assertGreaterOrEqual<uint32>("maxRules", maxRules, 1);
                return std::make_unique<IncrementalPredictor>(*this, maxRules, probabilityTransformationPtr_);
            }
    };

}
