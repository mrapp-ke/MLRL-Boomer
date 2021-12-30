/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/model/body.hpp"
#include "common/model/head.hpp"


/**
 * Defines an interface for all rules that consist of a body and a head.
 */
class IRule {

    public:

        virtual ~IRule() { };

        /**
         * Returns the body of the rule.
         *
         * @return A reference to an object of type `IBody` that represents the body of the rule
         */
        virtual const IBody& getBody() const = 0;

        /**
         * Returns the head of the rule.
         *
         * @return A reference to an object of type `IHead` that represents the head of the rule
         */
        virtual const IHead& getHead() const = 0;

        /**
         * Invokes some of the given visitor functions, depending on which ones are able to handle the rule's particular
         * type of body and head.
         *
         * @param emptyBodyVisitor          The visitor function for handling objects of the type `EmptyBody`
         * @param conjunctiveBodyVisitor    The visitor function for handling objects of the type `ConjunctiveBody`
         * @param completeHeadVisitor       The visitor function for handling objects of the type `CompleteHead`
         * @param partialHeadVisitor        The visitor function for handling objects of the type `PartialHead`
         */
        virtual void visit(IBody::EmptyBodyVisitor emptyBodyVisitor,
                           IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                           IHead::CompleteHeadVisitor completeHeadVisitor,
                           IHead::PartialHeadVisitor partialHeadVisitor) const = 0;

};
