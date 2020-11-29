/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "score_vector_label_wise_dense.h"

// Forward declarations
class AbstractEvaluatedPrediction;


/**
 * Defines an interface for all classes that process the scores that are stored by an `ILabelWiseScoreVector` in order
 * to convert them into the head of a rule, represented by an `AbstractEvaluatedPrediction`.
 */
class ILabelWiseScoreProcessor {

    public:

        virtual ~ILabelWiseScoreProcessor() { };

        /**
         * Processes the scores that are stored by a `DenseLabelWiseScoreVector` in order to convert them into the head
         * of a rule.
         *
         * @param bestHead      A pointer to an object of type `AbstractEvaluatedPrediction` that represents the best
         *                      head that has been created so far
         * @param scoreVector   A reference to an object of type `DenseLabelWiseScoreVector` that stores the scores to
         *                      be processed
         * @return              A pointer to an object of type `AbstractEvaluatedPrediction` that has been created or a
         *                      null pointer if no object has been created
         */
        virtual const AbstractEvaluatedPrediction* processScores(const AbstractEvaluatedPrediction* bestHead,
                                                                 const DenseLabelWiseScoreVector& scoreVector) = 0;

};
