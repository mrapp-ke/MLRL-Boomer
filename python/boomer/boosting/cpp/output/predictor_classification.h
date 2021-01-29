/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../../common/cpp/output/predictor.h"


namespace boosting {

    /**
     * Allows to predict the labels of given query examples using an existing rule-based model that has been learned
     * using a boosting algorithm.
     *
     * For prediction, the scores that are provided by the individual rules, are summed up. The aggregated scores are
     * then transformed into binary values according to a certain threshold (1 if a score exceeds the threshold, i.e.,
     * the label is relevant, 0 otherwise).
     */
    class ClassificationPredictor : public IPredictor<uint8> {

        private:

            float64 threshold_;

        public:

            /**
             * @param threshold The threshold to be used
             */
            ClassificationPredictor(float64 threshold);

            void predict(const CContiguousFeatureMatrix& featureMatrix, CContiguousView<uint8>& predictionMatrix,
                         const RuleModel& model) const override;

            void predict(const CsrFeatureMatrix& featureMatrix, CContiguousView<uint8>& predictionMatrix,
                         const RuleModel& model) const override;

    };

}
