/**
 * Provides classes that allow to post-process the predictions of rules once they have been learned by a boosting
 * algorithm.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/post_processing.h"


namespace boosting {

    /**
     * Post-processes the predictions of rules by shrinking their weights by a constant shrinkage parameter.
     */
    class ConstantShrinkageImpl : virtual public IPostProcessor {

        private:

            float64 shrinkage_;

        public:

            /**
             * @param shrinkage The shrinkage parameter. Must be in (0, 1).
             */
            ConstantShrinkageImpl(float64 shrinkage);

            void postProcess(AbstractPrediction& prediction) const override;

    };

}
