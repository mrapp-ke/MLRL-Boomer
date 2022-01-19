/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor_probability.hpp"


namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure a predictor that predicts label-wise probabilities
     * for given query examples, which estimate the chance of individual labels to be relevant, by summing up the scores
     * that are provided by individual rules of an existing rule-based models and transforming the aggregated scores
     * into probabilities in [0, 1] according to a certain transformation function that is applied to each label
     * individually.
     */
    class ILabelWiseProbabilityPredictorConfig {

        public:

            virtual ~ILabelWiseProbabilityPredictorConfig() { };

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
             * @return              A reference to an object of type `ILabelWiseProbabilityPredictorConfig` that
             *                      allows further configuration of the predictor
             */
            virtual ILabelWiseProbabilityPredictorConfig& setNumThreads(uint32 numThreads) = 0;

    };

    /**
     * Allows to configure a predictor that predicts label-wise probabilities for given query examples, which estimate
     * the chance of individual labels to be relevant, by summing up the scores that are provided by individual rules of
     * an existing rule-based models and transforming the aggregated scores into probabilities in [0, 1] according to a
     * certain transformation function that is applied to each label individually.
     */
    class LabelWiseProbabilityPredictorConfig final : public IProbabilityPredictorConfig,
                                                      public ILabelWiseProbabilityPredictorConfig {

        private:

            uint32 numThreads_;

        public:

            LabelWiseProbabilityPredictorConfig();

            uint32 getNumThreads() const override;

            ILabelWiseProbabilityPredictorConfig& setNumThreads(uint32 numThreads) override;

            std::unique_ptr<IProbabilityPredictorFactory> create() const override;

    };

}
