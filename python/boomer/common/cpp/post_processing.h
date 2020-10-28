/**
 * Provides classes that allow to post-process the predictions of rules once they have been learned.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "predictions.h"


/**
 * Defines an interface for all classes that allow to post-process the predictions of rules once they have been learned.
 */
class IPostProcessor {

    public:

        virtual ~IPostProcessor() { };

        /**
         * Post-processes the prediction of a rule.
         *
         * @param prediction A reference to an object of type `AbstractPrediction` that stores the predictions of a rule
         */
        virtual void postProcess(AbstractPrediction& prediction) const = 0;

};

/**
 * An implementation of the class `IPostProcessor` that does not perform any post-processing, but retains the original
 * predictions of rules.
 */
class NoPostProcessorImpl : virtual public IPostProcessor {

    public:

        void postProcess(AbstractPrediction& prediction) const override;

};
