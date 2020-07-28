/**
 * TODO
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/arrays.h"
#include "../../common/cpp/rule_evaluation.h"
#include "heuristics.h"


namespace rule_evaluation {

    /**
     * TODO
     */
    class CppLabelWiseRuleEvaluation {

        private:

            /**
             * TODO
             */
            heuristics::HeuristicFunction* heuristicFunction_;

        public:

            /**
             * TODO
             *
             * @param heuristicFunction
             */
            CppLabelWiseRuleEvaluation(heuristics::HeuristicFunction* heuristicFunction);

            /**
             * TODO
             *
             * @param labelIndices
             * @param minorityLabels
             * @param confusionMatricesTotal
             * @param confusionMatricesSubset
             * @param confusionMatricesCovered
             * @param uncovered
             * @param prediction
             */
            void calculateLabelWisePrediction(const intp* labelIndices, const uint8* minorityLabels,
                                              const float64* confusionMatricesTotal,
                                              const float64* confusionMatricesSubset,
                                              const float64* confusionMatricesCovered, bool uncovered,
                                              LabelWisePrediction* prediction);

    };

}
