/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "seco/statistics/statistics.hpp"
#include "seco/rule_evaluation/rule_evaluation_label_wise.hpp"


namespace seco {

    /**
     * Defines an interface for all classes that allow to store the elements of confusion matrices that are computed
     * independently for each label.
     */
    class ILabelWiseStatistics : public ICoverageStatistics {

        public:

            virtual ~ILabelWiseStatistics() { };

            /**
             * Sets the factory that allows to create instances of the class that is used for calculating the
             * predictions, as well as corresponding quality scores, of rules.
             *
             * @param ruleEvaluationFactory A reference to an object of type `ILabelWiseRuleEvaluationFactory` to be set
             */
            virtual void setRuleEvaluationFactory(const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory) = 0;

    };

}
