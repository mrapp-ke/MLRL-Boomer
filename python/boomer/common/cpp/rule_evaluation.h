/**
 * Provides classes that store the predictions of rules, as well as corresponding quality scores.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "predictions.h"
#include "input_data.h"


/**
 * An abstract base class for all classes that allow to calculate the predictions of a default rule.
 */
class AbstractDefaultRuleEvaluation {

    public:

        virtual ~AbstractDefaultRuleEvaluation();

        /**
         * Calculates the scores to be predicted by a default rule based on the ground truth label matrix.
         *
         * @param labelMatrix   A `LabelMatrix` that provides random access to the labels of the training examples
         * @return              A pointer to an object of type `Prediction`, representing the predictions of the default
         *                      rule
         */
        virtual Prediction* calculateDefaultPrediction(AbstractLabelMatrix* labelMatrix);

};
