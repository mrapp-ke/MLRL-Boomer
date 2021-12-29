/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor_regression.hpp"


namespace boosting {

    /**
     * Allows to create instances of the type `IRegressionPredictor` that allow to predict label-wise regression scores
     * for given query examples by summing up the scores that are provided by the individual rules of an existing
     * rule-based model for each label individually.
     */
    class LabelWiseRegressionPredictorFactory final : public IRegressionPredictorFactory {

        private:

            uint32 numThreads_;

        public:

            /**
             * @param numThreads The number of CPU threads to be used to make predictions for different query examples
             *                   in parallel. Must be at least 1
             */
            LabelWiseRegressionPredictorFactory(uint32 numThreads);

            std::unique_ptr<IRegressionPredictor> create(const RuleModel& model) const override;

    };

}
