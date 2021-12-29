/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor_regression.hpp"


namespace boosting {

    /**
     * An implementation of the type `IRegressionPredictor` that allows to predict label-wise regression scores for
     * given query examples by summing up the scores that are provided by the individual rules of an existing rule-based
     * model for each label individually.
     */
    class LabelWiseRegressionPredictor final : public IRegressionPredictor {

        private:

            uint32 numThreads_;

        public:

            /**
             * @param numThreads The number of CPU threads to be used to make predictions for different query examples
             *                   in parallel. Must be at least 1
             */
            LabelWiseRegressionPredictor(uint32 numThreads);

            void predict(const CContiguousFeatureMatrix& featureMatrix, CContiguousView<float64>& predictionMatrix,
                         const Rule& rule, const LabelVectorSet* labelVectors) const override;

            void predict(const CsrFeatureMatrix& featureMatrix, CContiguousView<float64>& predictionMatrix,
                         const Rule& rule, const LabelVectorSet* labelVectors) const override;

            void predict(const CContiguousFeatureMatrix& featureMatrix, CContiguousView<float64>& predictionMatrix,
                         const RuleModel& model, const LabelVectorSet* labelVectors) const override;

            void predict(const CsrFeatureMatrix& featureMatrix, CContiguousView<float64>& predictionMatrix,
                         const RuleModel& model, const LabelVectorSet* labelVectors) const override;

    };

}
