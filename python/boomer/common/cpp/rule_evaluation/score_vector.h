/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

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
         * Passes the scores to an `IScoreProcessor`.
         *
         * @param scoreProcessor A reference to an object of type `IScoreProcessor`, the scores should be passed to
         */
        virtual void processScores(IScoreProcessor& scoreProcessor) const = 0;

};
