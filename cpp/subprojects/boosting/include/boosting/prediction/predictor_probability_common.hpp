/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/prediction/predictor_score_common.hpp"
#include "boosting/prediction/transformation_probability.hpp"

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

}
