/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor_classification.hpp"


namespace seco {

    /**
     * Allows to configure a predictor that predicts whether individual labels of given query examples are relevant or
     * irrelevant by processing rules of an existing rule-based model in the order they have been learned. If a rule
     * covers an example, its prediction (1 if the label is relevant, 0 otherwise) is applied to each label
     * individually, if none of the previous rules has already predicted for a particular example and label.
     */
    class LabelWiseClassificationPredictorConfig final : public IClassificationPredictorConfig {

        private:

            uint32 numThreads_;

        public:

            LabelWiseClassificationPredictorConfig();

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
             * @return              A reference to an object of type `LabelWiseClassificationPredictorConfig` that
             *                      allows further configuration of the predictor
             */
            LabelWiseClassificationPredictorConfig& setNumThreads(uint32 numThreads);

    };

    /**
     * Allows to create instances of the class `IClassificationPredictor` that allow to predict whether individual
     * labels of given query examples are relevant or irrelevant by processing rules of an existing rule-based model in
     * the order they have been learned. If a rule covers an example, its prediction (1 if the label is relevant, 0
     * otherwise) is applied to each label individually, if none of the previous rules has already predicted for a
     * particular example and label.
     */
    class LabelWiseClassificationPredictorFactory final : public IClassificationPredictorFactory {

        private:

            uint32 numThreads_;

        public:

            /**
             * @param numThreads The number of CPU threads to be used to make predictions for different query examples
             *                   in parallel. Must be at least 1
             */
            LabelWiseClassificationPredictorFactory(uint32 numThreads);

            std::unique_ptr<IClassificationPredictor> create(const RuleList& model,
                                                             const LabelVectorSet* labelVectorSet) const override;

    };

}
