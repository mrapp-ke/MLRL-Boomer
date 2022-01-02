/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor_classification.hpp"
#include "common/measures/measure_similarity.hpp"


namespace boosting {

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

            std::unique_ptr<IClassificationPredictor> create(const RuleList& model,
                                                             const LabelVectorSet* labelVectors) const override;

    };

}
