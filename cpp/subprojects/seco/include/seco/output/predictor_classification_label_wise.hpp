/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor_classification.hpp"


namespace seco {

    /**
     * An implementation of the type `IClassificationPredictor` that allows to predict whether individual labels of
     * given query examples are relevant or irrelevant by processing rules of an existing rule-based model in the order
     * they have been learner. If a rule covers an example, its prediction (1 if the label is relevant, 0 otherwise) is
     * applied to each label individually, if none of the previous rules has already predicted for a particular example
     * and label.
     */
    class LabelWiseClassificationPredictor final : public IClassificationPredictor {

        private:

            uint32 numThreads_;

        public:

            /**
             * @param numThreads The number of CPU threads to be used to make predictions for different query examples
             *                   in parallel. Must be at least 1
             */
            LabelWiseClassificationPredictor(uint32 numThreads);

            void predict(const CContiguousFeatureMatrix& featureMatrix, CContiguousView<uint8>& predictionMatrix,
                         const RuleModel& model, const LabelVectorSet* labelVectors) const override;

            void predict(const CsrFeatureMatrix& featureMatrix, CContiguousView<uint8>& predictionMatrix,
                         const RuleModel& model, const LabelVectorSet* labelVectors) const override;

            std::unique_ptr<BinarySparsePredictionMatrix> predictSparse(
                const CContiguousFeatureMatrix& featureMatrix, uint32 numLabels, const RuleModel& model,
                const LabelVectorSet* labelVectors) const override;

            std::unique_ptr<BinarySparsePredictionMatrix> predictSparse(
                const CsrFeatureMatrix& featureMatrix, uint32 numLabels, const RuleModel& model,
                const LabelVectorSet* labelVectors) const override;

    };

}
