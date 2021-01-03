/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../data/matrix_dense.h"


/**
 * Type definition for a C-contiguous matrix that stores predictions.
 */
typedef DenseMatrix<float64> DensePredictionMatrix;

/**
 * Type definition for a C-contiguous boolean matrix.
 */
typedef DenseMatrix<uint8> Mask;

/**
 * Defines an interface for all classes that represent the head of a rule.
 */
class IHead {

    public:

        virtual ~IHead { };

        /**
         * Adds the scores that are contained by the head to a given vector of predictions.
         *
         * @param begin An iterator to the beginning of the predictions to be updated
         * @param end   An iterator to the end of the predictions to be updated
         */
        virtual void apply(DensePredictionMatrix::iterator begin, DensePredictionMatrix::iterator end) const = 0;

        /**
         * Adds the scores that are contained by the head to a given vector of predictions.
         *
         * The prediction is restricted to certain labels according to a given mask. The mask will be updated by this
         * function.
         *
         * @param predictionsBegin  An iterator to the beginning of the predictions to be updated
         * @param predictionsEnd    An iterator to the end of the predictions to be updated
         * @param maskBegin         An iterator to the beginning of the mask
         * @param maskEnd           An iterator to the end of the mask
         */
        virtual void apply(DensePredictionMatrix::iterator predictionsBegin,
                           DensePredictionMatrix::iterator predictionsEnd, Mask::iterator maskBegin,
                           Mask::iterator maskEnd) const = 0;

};
