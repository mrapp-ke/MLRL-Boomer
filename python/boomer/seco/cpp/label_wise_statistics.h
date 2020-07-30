/**
 * TODO
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/arrays.h"
#include "../../common/cpp/statistics.h"
#include "label_wise_rule_evaluation.h"


namespace statistics {

    /**
     * TODO
     */
    class LabelWiseRefinementSearchImpl {

        private:

            rule_evaluation::LabelWiseRuleEvaluationImpl* ruleEvaluation_;

            intp numLabels_;

            const intp* labelIndices_;

            AbstractLabelMatrix* labelMatrix_;

            const float64* uncoveredLabels_;

            const uint8* minorityLabels_;

            const float64* confusionMatricesTotal_;

            const float64* confusionMatricesSubset_;

            float64* confusionMatricesCovered_;

            float64* accumulatedConfusionMatricesCovered_;

            rule_evaluation::LabelWisePrediction* prediction_;

        public:

            /**
             * @param ruleEvaluation TODO
             * @param numLabels
             * @param labelIndices
             * @param labelMatrix
             * @param uncoveredLabels
             * @param minorityLabels
             * @param confusionMatricesTotal
             * @param confusionMatricesSubset
             */
            LabelWiseRefinementSearchImpl(rule_evaluation::LabelWiseRuleEvaluationImpl* ruleEvaluation,
                                          intp numLabels, const intp* labelIndices, AbstractLabelMatrix* labelMatrix,
                                          const float64* uncoveredLabels, const uint8* minorityLabels,
                                          const float64* confusionMatricesTotal,
                                          const float64* confusionMatricesSubset);

            ~LabelWiseRefinementSearchImpl();

            /**
             * TODO
             *
             * @param statisticsIndex
             * @param weight
             */
            void updateSearch(intp statisticIndex, uint32 weight);

            /**
             * TODO
             */
            void resetSearch();

            /**
             * TODO
             *
             * @param uncovered
             * @param accumulated
             */
            rule_evaluation::LabelWisePrediction* calculateLabelWisePrediction(bool uncovered, bool accumulated);

    };

}
