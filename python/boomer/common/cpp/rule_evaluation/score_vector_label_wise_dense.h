/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "score_vector_label_wise.h"
#include "score_vector_dense.h"


/**
 * An one-dimensional vector that stores the scores that may be predicted by a rule, as well as an overall quality
 * score that asses the overall quality of the rule, in a C-contiguous array.
 */
class DenseLabelWiseScoreVector : public DenseScoreVector, virtual public ILabelWiseScoreVector {

    private:

        DenseVector<float64> qualityScoreVector_;

    public:

        /**
         * @param numElements The number of labels for which the rule may predict
         */
        DenseLabelWiseScoreVector(uint32 numElements);

        typedef DenseVector<float64>::iterator quality_score_iterator;

        typedef DenseVector<float64>::const_iterator quality_score_const_iterator;

        /**
         * Returns a `quality_score_iterator` to the beginning of the quality scores.
         *
         * @return A `quality_score_iterator` to the beginning
         */
        quality_score_iterator quality_scores_begin();

        /**
         * Returns a `quality_score_iterator` to the end of the quality scores.
         *
         * @return A `quality_score_iterator` to the end
         */
        quality_score_iterator quality_scores_end();

        /**
         * Returns a `quality_score_const_iterator` to the beginning of the quality scores.
         *
         * @return A `quality_score_const_iterator` to the beginning
         */
        quality_score_const_iterator quality_scores_cbegin() const;

        /**
         * Returns a `quality_score_const_iterator` to the end of the quality scores.
         *
         * @return A `quality_score_const_iterator` to the end
         */
        quality_score_const_iterator quality_scores_cend() const;

        const AbstractEvaluatedPrediction* processScores(const AbstractEvaluatedPrediction* bestHead,
                                                         ILabelWiseScoreProcessor& scoreProcessor) const override;

};
