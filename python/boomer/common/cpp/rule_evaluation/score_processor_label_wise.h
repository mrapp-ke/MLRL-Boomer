/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "score_vector_label_wise_dense.h"


/**
 * Defines an interface for all classes that process the scores that are stored by an `ILabelWiseScoreVector`.
 */
class ILabelWiseScoreProcessor {

    public:

        virtual ~ILabelWiseScoreProcessor() { };

        /**
         * Processes the scores that are stored by a `DenseLabelWiseScoreVector`.
         *
         * @param scoreVector A reference to an object of type `DenseLabelWiseScoreVector` that stores the scores to be
         *                    processed
         */
        void processScores(const DenseLabelWiseScoreVector& scoreVector);

};
