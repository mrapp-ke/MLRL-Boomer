/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "score_vector_dense.h"

// Forward declarations
class AbstractEvaluatedPrediction;


/**
 * Defines an interface for all classes that process the scores that are stored by an `IScoreVector` in order to convert
 * them into the head of a rule, represented by an `AbstractEvaluatedPrediction`.
 */
class IScoreProcessor {

    public:

        virtual ~IScoreProcessor() { };

        /**
         * Processes the scores that are stored by a `DenseScoreVector` in order to convert them into the head of a
         * rule.
         *
         * @param bestHead      A pointer to an object of type `AbstractEvaluatedPrediction` that represents the best
         *                      head that has been created so far
         * @param scoreVector   A reference to an object of type `DenseScoreVector` that stores the scores to be
         *                      processed
         * @return              A reference to an object of type `AbstractEvaluatedPrediction` that represents the head
         *                      of a rule
         */
        virtual const AbstractEvaluatedPrediction& processScores(const AbstractEvaluatedPrediction* bestHead,
                                                                 const DenseScoreVector& scoreVector) = 0;

};
