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

        std::list<Rule> list_;

    public:

        typedef std::list<Rule>::const_iterator const_iterator;

        /**
         * Returns a `const_iterator` to the beginning of the rules.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the rules.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const;

        /**
         * Creates a new rule from a given body and head and adds it to the model.
         *
         * @param bodyPtr An unique pointer to an object of type `IBody` that should be used as the body of the rule
         * @param headPtr An unique pointer to an object of type `IHead` that should be used as the head of the rule
         */
        void addRule(std::unique_ptr<IBody> bodyPtr, std::unique_ptr<IHead> headPtr);

};
