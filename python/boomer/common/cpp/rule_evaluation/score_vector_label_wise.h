/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "score_vector.h"

// Forward declarations
class ILabelWiseScoreProcessor;


/**
 * Defines an interface for all one-dimensional vectors that store the scores that may be predicted by a rule, as well
 * as corresponding quality scores that assess the quality of the predictions for individual labels and an overall
 * quality score that assesses the overall quality of the rule.
 */
class ILabelWiseScoreVector : public IScoreVector {

    public:

        virtual ~ILabelWiseScoreVector() { };

        /**
         * Passes the scores to an `ILabelWiseScoreProcessor`.
         *
         * @param scoreProcessor A reference to an object of type `ILabelWiseScoreProcessor`, the scores should be
         *                       passed to
         */
        virtual void processScores(ILabelWiseScoreProcessor& scoreProcessor) const = 0;

};
