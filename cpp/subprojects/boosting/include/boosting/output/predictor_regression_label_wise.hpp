/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/output/predictor.hpp"


namespace boosting {

    /**
     * Allows to predict regression scores for given query examples using an existing rule-based model that has been
     * learned using a boosting algorithm.
     *
     * For prediction, the scores that are provided by the individual rules, are summed up for each label individually.
     */
    class LabelWiseRegressionPredictor : public IPredictor<float64> {

        private:

            uint32 numThreads_;

        public:

            /**
             * @param numThreads The number of CPU threads to be used to make predictions for different query examples
             *                   in parallel. Must be at least 1
             */
            LabelWiseRegressionPredictor(uint32 numThreads);

            void predict(const CContiguousFeatureMatrix& featureMatrix, CContiguousView<float64>& predictionMatrix,
                         const RuleModel& model) const override;

            void predict(const CsrFeatureMatrix& featureMatrix, CContiguousView<float64>& predictionMatrix,
                         const RuleModel& model) const override;

    };

}
