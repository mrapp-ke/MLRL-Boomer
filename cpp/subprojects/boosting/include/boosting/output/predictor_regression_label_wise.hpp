/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor_regression.hpp"


namespace boosting {

    /**
     * Allows to configure predictors that predict label-wise regression scores for given query examples by summing up
     * the scores that are provided by the individual rules of an existing rule-based model for each label individually.
     */
    class LabelWiseRegressionPredictorConfig final : public IRegressionPredictorConfig {

        private:

            uint32 numThreads_;

        public:

            LabelWiseRegressionPredictorConfig();

            /**
             * Returns the number of CPU threads that are used to make predictions for different query examples in
             * parallel.
             *
             * @return The number of CPU threads that are used to make predictions for different query examples in
             *         parallel
             */
            uint32 getNumThreads() const;

            /**
             * Sets the number of CPU threads that should be used to make predictions for different query examples in
             * parallel.
             *
             * @param numThreads    The number of CPU threads that should be used. Must be at least 1 or 0, if the
             *                      number of CPU threads should be chosen automatically
             * @return              A reference to an object of type `ExampleWiseClassificationPredictorConfig` that
             *                      allows further configuration of the predictor
             */
            LabelWiseRegressionPredictorConfig& setNumThreads(uint32 numThreads);

    };

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

            std::unique_ptr<IRegressionPredictor> create(const RuleList& model,
                                                         const LabelVectorSet* labelVectorSet) const override;

    };

}
