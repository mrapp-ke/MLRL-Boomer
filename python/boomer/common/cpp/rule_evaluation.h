/**
 * Implements classes for calculating the predictions of rules, as well as corresponding quality scores.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "data.h"


/**
 * Stores the scores that are predicted by a rule, as well as a quality score that assesses the overall quality of the
 * rule.
 */
class EvaluatedPrediction : public DenseVector<float64> {

    public:

        /**
         * @param numElements The number of labels for which the rule predicts
         */
        EvaluatedPrediction(uint32 numElements);

        /**
         * A score that assesses the overall quality of the prediction.
         */
        float64 overallQualityScore;

};

/**
 * Stores the scores that are predicted by a rule, as well as an overall quality score and a quality score for each
 * label.
 */
class LabelWiseEvaluatedPrediction : public EvaluatedPrediction {

    private:

        float64* qualityScores_;

    public:

        /**
         * @param numElements The number of labels for which the rule predicts
         */
        LabelWiseEvaluatedPrediction(uint32 numElements);

        ~LabelWiseEvaluatedPrediction();

        typedef float64* quality_score_iterator;

        typedef const float64* quality_score_const_iterator;

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

};
