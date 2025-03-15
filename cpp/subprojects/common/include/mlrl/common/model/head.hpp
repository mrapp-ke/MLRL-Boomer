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
         * A visitor function for handling objects of the type `CompleteHead`.
         *
         * @tparam ScoreType The type of the numerical scores that are stored by the head
         */
        template<typename ScoreType>
        using CompleteHeadVisitor = std::function<void(const CompleteHead<ScoreType>&)>;

        /**
         * A visitor function for handling objects of the type `PartialHead`.
         *
         * @tparam ScoreType The type of the numerical scores that are stored by the head
         */
        template<typename ScoreType>
        using PartialHeadVisitor = std::function<void(const PartialHead<ScoreType>&)>;

        /**
         * Invokes one of the given visitor functions, depending on which one is able to handle this particular type of
         * head.
         *
         * @param complete32BitHeadVisitor  A visitor function for handling objects of the type `CompleteHead<float32>`
         * @param complete64BitHeadVisitor  A visitor function for handling objects of the type `CompleteHead<float64>`
         * @param partial32BitHeadVisitor   A visitor function for handling objects of the type `PartialHead<float32>`
         * @param partial64BitHeadVisitor   A visitor function for handling objects of the type `PartialHead<float64>`
         */
        virtual void visit(CompleteHeadVisitor<float32> complete32BitHeadVisitor,
                           CompleteHeadVisitor<float64> complete64BitHeadVisitor,
                           PartialHeadVisitor<float32> partial32BitHeadVisitor,
                           PartialHeadVisitor<float64> partial64BitHeadVisitor) const = 0;
};
