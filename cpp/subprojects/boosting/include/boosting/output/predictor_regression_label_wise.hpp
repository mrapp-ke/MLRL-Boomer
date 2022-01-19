/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor_regression.hpp"


namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure predictors that predict label-wise regression scores
     * for given query examples by summing up the scores that are provided by the individual rules of an existing
     * rule-based model for each label individually.
     */
    class ILabelWiseRegressionPredictorConfig {

        public:

            virtual ~ILabelWiseRegressionPredictorConfig() { };

            /**
             * Returns the number of CPU threads that are used to make predictions for different query examples in
             * parallel.
             *
             * @return The number of CPU threads that are used to make predictions for different query examples in
             *         parallel
             */
            virtual uint32 getNumThreads() const = 0;

            /**
             * Sets the number of CPU threads that should be used to make predictions for different query examples in
             * parallel.
             *
             * @param numThreads    The number of CPU threads that should be used. Must be at least 1 or 0, if the
             *                      number of CPU threads should be chosen automatically
             * @return              A reference to an object of type `ILabelWiseRegressionPredictorConfig` that allows
             *                      further configuration of the predictor
             */
            virtual ILabelWiseRegressionPredictorConfig& setNumThreads(uint32 numThreads) = 0;

    };

    /**
     * Allows to configure predictors that predict label-wise regression scores for given query examples by summing up
     * the scores that are provided by the individual rules of an existing rule-based model for each label individually.
     */
    class LabelWiseRegressionPredictorConfig final : public IRegressionPredictorConfig,
                                                     public ILabelWiseRegressionPredictorConfig {

        private:

            uint32 numThreads_;

        public:

            LabelWiseRegressionPredictorConfig();

            uint32 getNumThreads() const override;

            ILabelWiseRegressionPredictorConfig& setNumThreads(uint32 numThreads) override;

            std::unique_ptr<IRegressionPredictorFactory> create() const override;

    };

}
