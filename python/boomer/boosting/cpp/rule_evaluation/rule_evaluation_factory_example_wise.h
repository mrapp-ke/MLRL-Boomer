/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "rule_evaluation_example_wise.h"
#include <memory>


namespace boosting {

    /**
     * Defines an interface for all factories that allow to create instances of the type `IExampleWiseRuleEvaluation`.
     */
    class IExampleWiseRuleEvaluationFactory {

        public:

            virtual ~IExampleWiseRuleEvaluationFactory() { };

            /**
             * Creates and returns a new object of type `ILabelWiseRuleEvaluation` that allows to calculate the
             * predictions of rules that predict for all available labels.
             *
             * @param indexVector   A reference to an object of type `FullIndexVector` that provides access to the
             *                      indices of the labels for which the rules may predict
             * @return              An unique pointer to an object of type `ILabelWiseRuleEvaluation` that has been
             *                      created
             */
            virtual std::unique_ptr<IExampleWiseRuleEvaluation> create(const FullIndexVector& indexVector) const = 0;

            /**
             * Creates and returns a new object of type `ILabelWiseRuleEvaluation` that allows to calculate the
             * predictions of rules that predict for a subset of the available labels.
             *
             * @param indexVector   A reference to an object of type `PartialIndexVector` that provides access to the
             *                      indices of the labels for which the rules may predict
             * @return              An unique pointer to an object of type `ILabelWiseRuleEvaluation` that has been
             *                      created
             */
            virtual std::unique_ptr<IExampleWiseRuleEvaluation> create(const PartialIndexVector& indexVector) const = 0;

    };

}
