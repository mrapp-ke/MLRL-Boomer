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

            /**
             * TODO
             */
            rule_evaluation::LabelWiseRuleEvaluationImpl* ruleEvaluation_;

            /**
             * TODO
             */
            intp numLabels_;

            /**
             * TODO
             */
            const intp* labelIndices_;

            /**
             * TODO
             */
            AbstractLabelMatrix* labelMatrix_;

            /**
             * TODO
             */
            const float64* uncoveredLabels_;

            /**
             * TODO
             */
            const uint8* minorityLabels_;

            /**
             * TODO
             */
            const float64* confusionMatricesTotal_;

            /**
             * TODO
             */
            const float64* confusionMatricesSubset_;

            /**
             * TODO
             */
            rule_evaluation::LabelWisePrediction* prediction_;

        public:

            /**
             * TODO
             *
             * @param ruleEvaluation
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

            /**
             * TODO
             */
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
