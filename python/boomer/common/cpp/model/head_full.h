/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "head.h"
#include "../head_refinement/prediction_full.h"


/**
 * A head that contains a numerical score for each available label.
 */
class FullHead final : public IHead {

    private:

        uint32 numScores_;

        float64* scores_;

    public:

        /**
         * @param prediction A reference to an object of type `FullPrediction` that stores the scores to be contained by
         *                   the head
         */
        FullHead(const FullPrediction& prediction);

        ~FullHead();

        void apply(DensePredictionMatrix::iterator begin, DensePredictionMatrix::iterator end) const override;

        void apply(DensePredictionMatrix::iterator predictionsBegin, DensePredictionMatrix::iterator predictionsEnd,
                   Mask::iterator maskBegin, Mask::iterator maskEnd) const override;

};
