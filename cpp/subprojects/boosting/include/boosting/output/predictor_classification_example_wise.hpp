/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor_classification.hpp"
#include "common/measures/measure_similarity.hpp"


namespace boosting {

    /**
     * An implementation of the type `IExampleWiseClassificationPredictor` that allows to predict known label vectors
     * for given query examples by summing up the scores that are provided by an existing rule-based model and comparing
     * the aggregated score vector to the known label vectors according to a certain distance measure. The label vector
     * that is closest to the aggregated score vector is finally predicted.
     */
    class ExampleWiseClassificationPredictor final : public IClassificationPredictor {

        private:

            std::unique_ptr<ISimilarityMeasure> similarityMeasurePtr_;

            uint32 numThreads_;

        public:

            /**
             * @param similarityMeasureFactoryPtr   An unique pointer to an object of type `ISimilarityMeasure` that
             *                                      implements the similarity measure that should be used to quantify
             *                                      the similarity between predictions and known label vectors
             * @param numThreads                    The number of CPU threads to be used to make predictions for
             *                                      different query examples in parallel. Must be at least 1
             */
            ExampleWiseClassificationPredictor(std::unique_ptr<ISimilarityMeasure> similarityMeasurePtr,
                                               uint32 numThreads);

            /**
             * Obtains predictions for different examples, based on predicted scores, and writes them to a given
             * prediction matrix.
             *
             * @param scoreMatrix       A reference to an object of type `CContiguousConstView` that stores the
             *                          predicted scores
             * @param predictionMatrix  A reference to an object of type `CContiguousView`, the predictions should be
             *                          written to. May contain arbitrary values
             * @param labelVectors      A pointer to an object of type `LabelVectorSet` that stores all known label
             *                          vectors or a null pointer, if no such set is available
             */
            // TODO Move to interface IClassificationPredictor
            void transform(const CContiguousConstView<float64>& scoreMatrix, CContiguousView<uint8>& predictionMatrix,
                           const LabelVectorSet* labelVectors) const;

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

    /**
     * Allows to create instances of the type `IClassificationPredictor` that allow to predict known label vectors for
     * given query examples by summing up the scores that are provided by an existing rule-based model and comparing the
     * aggregated score vector to the known label vectors according to a certain distance measure. The label vector that
     * is closest to the aggregated score vector is finally predicted.
     */
    class ExampleWiseClassificationPredictorFactory final : public IClassificationPredictorFactory {

        private:

            std::unique_ptr<ISimilarityMeasureFactory> similarityMeasureFactoryPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param similarityMeasureFactoryPtr   An unique pointer to an object of type `ISimilarityMeasureFactory`
             *                                      that allows to create implementations of the similarity measure
             *                                      that should be used to quantify the similarity between predictions
             *                                      and known label vectors
             * @param numThreads                    The number of CPU threads to be used to make predictions for
             *                                      different query examples in parallel. Must be at least 1
             */
            ExampleWiseClassificationPredictorFactory(
                std::unique_ptr<ISimilarityMeasureFactory> similarityMeasureFactoryPtr, uint32 numThreads);

            std::unique_ptr<IClassificationPredictor> create(const RuleModel& model) const override;

    };

}
