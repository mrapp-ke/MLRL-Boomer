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

        uint32 numElements_;

        float64* scores_;

    public:

        /**
         * @param prediction A reference to an object of type `FullPrediction` that stores the scores to be contained by
         *                   the head
         */
        FullHead(const FullPrediction& prediction);

        ~FullHead();

        typedef const float64* score_const_iterator;

        /**
         * Returns the number of scores that are contained by the head.
         *
         * @return The number of scores
         */
        uint32 getNumElements() const;

        /**
         * Returns a `score_const_iterator` to the beginning of the scores that are contained by the head.
         *
         * @return A `score_const_iterator` to the beginning
         */
        score_const_iterator scores_cbegin() const;

        /**
         * Returns a `score_const_iterator` to the end of the scores that are contained by the head.
         *
         * @return A `score_const_iterator` to the end
         */
        score_const_iterator scores_cend() const;

        void visit(FullHeadVisitor fullHeadVisitor, PartialHeadVisitor partialHeadVisitor) const override;

};
