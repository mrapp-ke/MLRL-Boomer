/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/output/predictor.hpp"
#include "common/input/label_vector_set.hpp"
#include "common/measures/measure_similarity.hpp"
#include <functional>


namespace boosting {

    /**
     * Allows to predict the labels of given query examples using an existing rule-based model that has been learned
     * using a boosting algorithm.
     *
     * For prediction, the scores that are provided by the individual rules, are summed up. For each query example, the
     * aggregated score vector is then compared to known label sets in order to obtain a distance measure. The label set
     * that is closest to the aggregated score vector is finally predicted.
     */
    class ExampleWiseClassificationPredictor : public IPredictor<uint8> {

        private:

            LabelVectorSet<uint32> labelVectors_;

            std::shared_ptr<ISimilarityMeasure> measurePtr_;

            uint32 numThreads_;

        public:

            /**
             * @param measurePtr    A shared pointer to an object of type `ISimilarityMeasure` that should be used to
             *                      quantify the similarity between predictions and known label vectors
             * @param numThreads    The number of CPU threads to be used to make predictions for different query
             *                      examples in parallel. Must be at least 1
             */
            ExampleWiseClassificationPredictor(std::shared_ptr<ISimilarityMeasure> measurePtr, uint32 numThreads);

            /**
             * A visitor function for handling objects of the type `LabelVector`.
             */
            typedef std::function<void(const LabelVector&)> LabelVectorVisitor;

            /**
             * Adds a known label vector that may be predicted for individual query examples.
             *
             * @param labelVectorPtr An unique pointer to an object of type `LabelVector`
             */
            void addLabelVector(std::unique_ptr<LabelVector> labelVectorPtr);

            /**
             * Invokes the given visitor function for each unique label vector that has been provided via the function `addLabelVector`.
             *
             * @param visitor The visitor function for handling objects of the type `LabelVector`
             */
            void visit(LabelVectorVisitor visitor) const;

            /**
             * Obtains predictions for different examples, based on predicted scores, and writes them to a given
             * prediction matrix.
             *
             * @param scoreMatrix       A reference to an object of type `CContiguousConstView` that stores the 
             *                          predicted scores
             * @param predictionMatrix  A reference to an object of type `CContiguousView`, the predictions should be
             *                          written to. May contain arbitrary values
             */
            void transform(const CContiguousConstView<float64>& scoreMatrix,
                           CContiguousView<uint8>& predictionMatrix) const;

            void predict(const CContiguousFeatureMatrix& featureMatrix, CContiguousView<uint8>& predictionMatrix,
                         const RuleModel& model) const override;

            void predict(const CsrFeatureMatrix& featureMatrix, CContiguousView<uint8>& predictionMatrix,
                         const RuleModel& model) const override;

    };

}
