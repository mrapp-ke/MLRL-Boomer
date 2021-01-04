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

        void apply(CContiguousView<float64>::iterator begin, CContiguousView<float64>::iterator end) const override;

        void apply(CContiguousView<float64>::iterator predictionsBegin,
                   CContiguousView<float64>::iterator predictionsEnd, PredictionMask::iterator maskBegin,
                   PredictionMask::iterator maskEnd) const override;

};
