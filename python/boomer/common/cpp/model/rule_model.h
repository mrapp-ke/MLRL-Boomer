/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../../common/cpp/model/rule.h"
#include <list>


/**
 * A model that stores several rules in a list.
 */
class RuleModel final {

    private:

        std::list<std::unique_ptr<Rule>> list_;

    public:

        /**
         * Adds a new rule to the model.
         *
         * @param rulePtr An unique pointer to an object of type `Rule` that should be added
         */
        void addRule(std::unique_ptr<Rule> rulePtr);

};
