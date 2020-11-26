/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "score_vector_dense.h"


/**
 * Defines an interface for all classes that process the scores that are stored by an `IScoreVector`.
 */
class IScoreProcessor {

    public:

        virtual ~IScoreProcessor() { };

        /**
         * Processes the scores that are stored by a `DenseScoreVector`.
         *
         * @param scoreVector A reference to an object of type `DenseScoreVector` that stores the scores to be processed
         */
        void processScores(const DenseScoreVector& scoreVector);

};
