/**
 * Implements classes for calculating the predictions of rules, as well as corresponding quality scores.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "data/vector_dense.h"


/**
 * Stores the scores that are predicted by a rule, as well as a quality score that assesses the overall quality of the
 * rule.
 */
class EvaluatedPrediction {

    private:

        DenseVector<float64> predictedScoreVector_;

    public:

        /**
         * @param numElements The number of labels for which the rule predicts
         */
        EvaluatedPrediction(uint32 numElements);

        typedef DenseVector<float64>::iterator score_iterator;

        typedef DenseVector<float64>::const_iterator score_const_iterator;

        /**
         * Returns a `score_iterator` to the beginning of the predicted scores.
         *
         * @return A `score_iterator` to the beginning
         */
        score_iterator scores_begin();

        /**
         * Returns a `score_iterator` to the end of the predicted scores.
         *
         * @return A `score_iterator` to the end
         */
        score_iterator scores_end();

        /**
         * Returns a `score_const_iterator` to the beginning of the predicted scores.
         *
         * @return A `score_const_iterator` to the beginning
         */
        score_const_iterator scores_cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the predicted scores.
         *
         * @return A `const_iterator` to the end
         */
        score_const_iterator scores_cend() const;

        /**
         * Returns the number of labels for which the rule predicts.
         *
         * @return The number of labels for which the rule predict
         */
        uint32 getNumElements() const;

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

        DenseVector<float64> qualityScoreVector_;

    public:

        /**
         * @param numElements The number of labels for which the rule predicts
         */
        LabelWiseEvaluatedPrediction(uint32 numElements);

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

};
