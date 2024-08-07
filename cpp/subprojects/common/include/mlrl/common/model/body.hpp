/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view.hpp"

#include <functional>

// Forward declarations
class EmptyBody;
class ConjunctiveBody;

/**
 * Defines an interface for all classes that allow to check whether an example is covered or not.
 */
class MLRLCOMMON_API IConditional {
    public:

        virtual ~IConditional() {}

        /**
         * Returns whether an individual example, which is stored in a C-contiguous matrix, is covered or not.
         *
         * @param begin An iterator to the beginning of the example's feature values
         * @param end   An iterator to the end of the example's feature values
         * @return      True, if the example is covered, false otherwise
         */
        virtual bool covers(View<const float32>::const_iterator begin,
                            View<const float32>::const_iterator end) const = 0;

        /**
         * Returns whether an individual example, which is stored in a CSR sparse matrix, is covered or not.
         *
         * @param indicesBegin  An iterator to the beginning of the example's feature values
         * @param indicesEnd    An iterator to the end of the example's feature values
         * @param valuesBegin   An iterator to the beginning of the example's feature_indices
         * @param valuesEnd     An iterator to the end of the example's feature indices
         * @param sparseValue   The value that should be used for sparse feature values
         * @param tmpArray1     An iterator that is used to temporarily store dense feature values. May contain
                                arbitrary values
         * @param tmpArray2     An iterator that is used to temporarily keep track of the feature indices with dense
                                feature values. Must not contain any elements with value `n`
         * @param n             An arbitrary number. If this function is called multiple times for different examples,
         *                      but using the same `tmpArray2`, the number must be unique for each of the function
         *                      invocations
         * @return              True, if the example is covered, false otherwise
         */
        virtual bool covers(View<uint32>::const_iterator indicesBegin, View<uint32>::const_iterator indicesEnd,
                            View<float32>::const_iterator valuesBegin, View<float32>::const_iterator valuesEnd,
                            float32 sparseValue, View<float32>::iterator tmpArray1, View<uint32>::iterator tmpArray2,
                            uint32 n) const = 0;
};

/**
 * Defines an interface for all classes that represent the body of a rule.
 */
class MLRLCOMMON_API IBody : public IConditional {
    public:

        virtual ~IBody() override {}

        /**
         * A visitor function for handling objects of the type `EmptyBody`.
         */
        typedef std::function<void(const EmptyBody&)> EmptyBodyVisitor;

        /**
         * A visitor function for handling objects of the type `ConjunctiveBody`.
         */
        typedef std::function<void(const ConjunctiveBody&)> ConjunctiveBodyVisitor;

        /**
         * Invokes one of the given visitor functions, depending on which one is able to handle this particular type of
         * body.
         *
         * @param emptyBodyVisitor          The visitor function for handling objects of the type `EmptyBody`
         * @param conjunctiveBodyVisitor    The visitor function for handling objects of the type `ConjunctiveBody`
         */
        virtual void visit(EmptyBodyVisitor emptyBodyVisitor, ConjunctiveBodyVisitor conjunctiveBodyVisitor) const = 0;
};
