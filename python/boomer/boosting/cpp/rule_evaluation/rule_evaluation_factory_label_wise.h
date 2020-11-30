/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "rule_evaluation_label_wise.h"
#include "../../../common/cpp/indices/index_vector_full.h"
#include "../../../common/cpp/indices/index_vector_partial.h"
#include <memory>


namespace boosting {

    /**
     * Defines an interface for all factories that allow to create instances of the type `ILabelWiseRuleEvaluation`.
     */
    class ILabelWiseRuleEvaluationFactory {

        public:

            virtual ~ILabelWiseRuleEvaluationFactory() { };

            /**
             * Creates a new instance of the class `ILabelWiseRuleEvaluation` that allows to calculate the predictions
             * of rules that predict for all available labels.
             *
             * @param indexVector   A reference to an object of the type `FullIndexVector` that provides access to the
             *                      indices of the labels for which the rules may predict
             * @return              An unique pointer to an object of type `ILabelWiseRuleEvaluation` that has been
             *                      created
             */
            virtual std::unique_ptr<ILabelWiseRuleEvaluation> create(const FullIndexVector& indexVector) const = 0;

            /**
             * Creates a new instance of the class `ILabelWiseRuleEvaluation` that allows to calculate the predictions
             * of rules that predict for a subset of the available labels.
             *
             * @param indexVector   A reference to an object of the type `PartialIndexVector` that provides access to
             *                      the indices of the labels for which the rules may predict
             * @return              An unique pointer to an object of type `ILabelWiseRuleEvaluation` that has been
             *                      created
             */
            virtual std::unique_ptr<ILabelWiseRuleEvaluation> create(const PartialIndexVector& indexVector) const = 0;


    };

}
