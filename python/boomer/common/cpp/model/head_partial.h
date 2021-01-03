/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "head.h"
#include "../head_refinement/prediction_partial.h"


/**
 * A head that contains a numerical score for a subset of the available labels.
 */
class PartialHead final : public IHead {

    private:

        uint32 numScores_;

        float64* scores_;

        uint32* labelIndices_;

    public:

        /**
         * @param prediction A reference to an object of type `PartialPrediction` that stores the scores to be contained
         *                   by the head
         */
        PartialHead(const PartialPrediction& prediction);

        ~PartialHead();

        void apply(DensePredictionMatrix::iterator begin, DensePredictionMatrix::iterator end) const override;

        void apply(DensePredictionMatrix::iterator predictionsBegin, DensePredictionMatrix::iterator predictionsEnd,
                   Mask::iterator maskBegin, Mask::iterator maskEnd) const override;

};
