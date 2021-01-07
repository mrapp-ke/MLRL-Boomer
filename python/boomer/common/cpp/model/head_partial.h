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

        uint32 numElements_;

        float64* scores_;

        uint32* labelIndices_;

    public:

        /**
         * @param prediction A reference to an object of type `PartialPrediction` that stores the scores to be contained
         *                   by the head
         */
        PartialHead(const PartialPrediction& prediction);

        ~PartialHead();

        typedef const float64* score_const_iterator;

        typedef const uint32* index_const_iterator;

        /**
         * Returns the number of numerical scores that are contained by the head.
         *
         * @return The number of numerical scores
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

        /**
         * Returns an `index_const_iterator` to the beginning of the indices, the scores that are contained by the head
         * correspond to.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the indices, the scores that are contained by the head,
         * correspond to.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const;

        void apply(CContiguousView<float64>::iterator begin, CContiguousView<float64>::iterator end) const override;

        void apply(CContiguousView<float64>::iterator predictionsBegin,
                   CContiguousView<float64>::iterator predictionsEnd, PredictionMask::iterator maskBegin,
                   PredictionMask::iterator maskEnd) const override;

        void visit(IHead::FullHeadVisitor fullHeadVisitor, IHead::PartialHeadVisitor partialHeadVisitor) const override;

};
