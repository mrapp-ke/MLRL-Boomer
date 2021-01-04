/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../data/matrix_dense.h"

typedef DenseMatrix<uint8> PredictionMask;


/**
 * Defines an interface for all classes that represent the head of a rule.
 */
class IHead {

    public:

        virtual ~IHead() { };

        /**
         * Adds the scores that are contained by the head to a given vector of predictions.
         *
         * @param begin An iterator to the beginning of the predictions to be updated
         * @param end   An iterator to the end of the predictions to be updated
         */
        virtual void apply(DenseMatrix<float64>::iterator begin, DenseMatrix<float64>::iterator end) const = 0;

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
        virtual void apply(DenseMatrix<float64>::iterator predictionsBegin,
                           DenseMatrix<float64>::iterator predictionsEnd, PredictionMask::iterator maskBegin,
                           PredictionMask::iterator maskEnd) const = 0;

};
