/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../head_refinement/prediction_evaluated.h"

// Forward declarations
class IScoreProcessor;


/**
 * Defines an interface for all one-dimensional vectors that store the scores that may be predicted by a rule, as well
 * as a quality score that assess the overall quality of the rule.
 */
class IScoreVector {

    public:

        virtual ~IScoreVector() { };

        /**
         * Passes the scores to an `IScoreProcessor` in order to convert into the head of a rule.
         *
         * @param bestHead       A reference to an object of type `AbstractEvaluatedPrediction`, representing the best
         *                       head that has been created so far
         * @param scoreProcessor A reference to an object of type `IScoreProcessor`, the scores should be passed to
         * @return               A reference to an object of type `AbstractEvaluatedPrediction` that has been created
         */
        virtual const AbstractEvaluatedPrediction& processScores(const AbstractEvaluatedPrediction* bestHead,
                                                                 IScoreProcessor& scoreProcessor) const = 0;

};
