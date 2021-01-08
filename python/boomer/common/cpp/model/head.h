/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../data/matrix_dense.h"
#include "../data/view_c_contiguous.h"
#include <functional>

// Forward declarations
class FullHead;
class PartialHead;

typedef DenseMatrix<uint8> PredictionMask;

/**
 * Defines an interface for all classes that represent the head of a rule.
 */
class IHead {

    public:

        virtual ~IHead() { };

        typedef std::function<void(const FullHead& head)> FullHeadVisitor;

        typedef std::function<void(const PartialHead& head)> PartialHeadVisitor;

        /**
         * Adds the scores that are contained by the head to a given vector of predictions.
         *
         * @param begin An iterator to the beginning of the predictions to be updated
         * @param end   An iterator to the end of the predictions to be updated
         */
        virtual void apply(CContiguousView<float64>::iterator begin, CContiguousView<float64>::iterator end) const = 0;

        /**
         * Adds the scores that are contained by the head to a given vector of predictions.
         *
         * The prediction is restricted to labels for which the corresponding element in the given mask is zero, i.e.,
         * for which no rule has predicted yet. The mask will be updated by this function by setting all elements for
         * which a prediction has been made to a non-zero value.
         *
         * @param predictionsBegin  An iterator to the beginning of the predictions to be updated
         * @param predictionsEnd    An iterator to the end of the predictions to be updated
         * @param maskBegin         An iterator to the beginning of the mask
         * @param maskEnd           An iterator to the end of the mask
         */
        virtual void apply(CContiguousView<float64>::iterator predictionsBegin,
                           CContiguousView<float64>::iterator predictionsEnd, PredictionMask::iterator maskBegin,
                           PredictionMask::iterator maskEnd) const = 0;

        /**
         * Invokes one of the given visitor functions, depending on which one is able to handle this particular type of
         * head.
         *
         * @param fullHeadVisitor       The visitor function for handling objects of the type `FullHead`
         * @param partialHeadVisitor    The visitor function for handling objects of the type `PartialHead`
         */
        virtual void visit(FullHeadVisitor fullHeadVisitor, PartialHeadVisitor partialHeadVisitor) const = 0;

};
