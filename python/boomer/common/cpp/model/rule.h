/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "body.h"
#include "head.h"
#include <memory>


/**
 * A rule that consists of a body and a head.
 */
class Rule final {

    private:

        std::unique_ptr<IBody> bodyPtr_;

        std::unique_ptr<IHead> headPtr_;

    public:

        /**
         * @param bodyPtr   An unique pointer to an object of type `IBody` that represents the body of the rule
         * @param headPtr   An unique pointer to an object of type `IHead` that represents the head of the rule
         */
        Rule(std::unique_ptr<IBody> bodyPtr, std::unique_ptr<IHead> headPtr);

        /**
         * Returns the body of the rule.
         *
         * @return A reference to an object of type `IBody` that represents the body of the rule
         */
        const IBody& getBody() const;

        /**
         * Returns the head of the rule.
         *
         * @return A reference to an object of type `IHead` that represents the head of the rule
         */
        const IHead& getHead() const;

};
