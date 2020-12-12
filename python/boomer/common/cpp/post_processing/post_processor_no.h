/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "post_processor.h"


/**
 * An implementation of the class `IPostProcessor` that does not perform any post-processing, but retains the original
 * predictions of rules.
 */
class NoPostProcessor final : public IPostProcessor {

    public:

        void postProcess(AbstractPrediction& prediction) const override;

};
