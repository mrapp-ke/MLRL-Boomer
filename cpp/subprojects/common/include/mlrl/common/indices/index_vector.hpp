/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

#include <functional>
#include <memory>

// Forward declarations
class PartialIndexVector;
class CompleteIndexVector;

/**
 * Defines an interface for all classes that provide random access to indices.
 */
class IIndexVector {
    public:

        virtual ~IIndexVector() {}

        /**
         * A visitor function for handling objects of the type `PartialIndexVector`.
         */
        typedef std::function<void(const PartialIndexVector&)> PartialIndexVectorVisitor;

        /**
         * A visitor function for handling objects of the type `CompleteIndexVector`.
         */
        typedef std::function<void(const CompleteIndexVector&)> CompleteIndexVectorVisitor;

        /**
         * Returns the number of indices.
         *
         * @return The number of indices
         */
        virtual uint32 getNumElements() const = 0;

        /**
         * Returns whether the indices are partial, i.e., some indices in the range [0, getNumElements()) are missing,
         * or not.
         *
         * @return True, if the indices are partial, false otherwise
         */
        virtual bool isPartial() const = 0;

        /**
         * Returns the index at a specific position.
         *
         * @param pos   The position of the index. Must be in [0, getNumElements())
         * @return      The index at the given position
         */
        virtual uint32 getIndex(uint32 pos) const = 0;

        /**
         * Invokes one of the given visitor functions, depending on which one is able to handle this particular type of
         * vector.
         *
         * @param partialIndexVectorVisitor     The visitor function for handling objects of the type
         *                                      `PartialIndexVector`
         * @param completeIndexVectorVisitor    The visitor function for handling objects of the type
         *                                      `CompleteIndexVector`
         */
        virtual void visit(PartialIndexVectorVisitor partialIndexVectorVisitor,
                           CompleteIndexVectorVisitor completeIndexVectorVisitor) const = 0;
};
