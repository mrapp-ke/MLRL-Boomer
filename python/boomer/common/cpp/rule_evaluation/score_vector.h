/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../data/types.h"

// Forward declarations
class AbstractPrediction;

/**
 * Defines an interface for all one-dimensional vectors that store the scores that may be predicted by a rule, as well
 * as a quality score that assess the overall quality of the rule.
 */
class IScoreVector {

    public:

        virtual ~IScoreVector() { };

        /**
         * A score that assesses the overall quality of the predicted score.
         */
        float64 overallQualityScore;

        /**
         * Sets the scores of a specific prediction to the scores that are stored in this vector.
         *
         * @param prediction A reference to an object of type `AbstractPrediction` that should be updated
         */
        virtual void updatePrediction(AbstractPrediction& prediction) const = 0;

};
