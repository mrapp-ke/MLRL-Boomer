/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include <functional>

// Forward declarations
template<typename ScoreType>
class CompleteHead;

template<typename ScoreType>
class PartialHead;

/**
 * Defines an interface for all classes that represent the head of a rule.
 */
class MLRLCOMMON_API IHead {
    public:

        virtual ~IHead() {}

        /**
         * A visitor function for handling objects of the type `CompleteHead<float64>`.
         */
        typedef std::function<void(const CompleteHead<float64>&)> CompleteHeadVisitor;

        /**
         * A visitor function for handling objects of the type `PartialHead<float64>`.
         */
        typedef std::function<void(const PartialHead<float64>&)> PartialHeadVisitor;

        /**
         * Invokes one of the given visitor functions, depending on which one is able to handle this particular type of
         * head.
         *
         * @param completeHeadVisitor   The visitor function for handling objects of the type `CompleteHead<float64>`
         * @param partialHeadVisitor    The visitor function for handling objects of the type `PartialHead<float64>`
         */
        virtual void visit(CompleteHeadVisitor completeHeadVisitor, PartialHeadVisitor partialHeadVisitor) const = 0;
};
