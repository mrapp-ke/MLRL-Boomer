/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor_classification.hpp"


namespace boosting {

    /**
     * An implementation of the type `IClassificationPredictor` that allows to predict whether individual labels of
     * given query examples are relevant or irrelevant by summing up the scores that are provided by the individual
     * rules of an existing rule-based model and transforming them into binary values according to a certain threshold
     * that is applied to each label individually (1 if a score exceeds the threshold, i.e., the label is relevant, 0
     * otherwise).
     */
    class LabelWiseClassificationPredictor final : public IClassificationPredictor {

        private:

            float64 threshold_;

            uint32 numThreads_;

        public:

            /**
             * @param threshold     The threshold to be used
             * @param numThreads    The number of CPU threads to be used to make predictions for different query
             *                      examples in parallel. Must be at least 1
             */
            LabelWiseClassificationPredictor(float64 threshold, uint32 numThreads);

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
