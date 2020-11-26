/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "score_vector.h"
#include "../data/vector_dense.h"


/**
 * An one-dimensional vector that stores the scores that may be predicted by a rule, as well as an overall quality
 * score that asses the overall quality of the rule, in a C-contiguous array.
 */
class DenseScoreVector : virtual public IScoreVector {

    private:

        DenseVector<float64> predictedScoreVector_;

    public:

        /**
         * @param numElements The number of labels for which the rule may predict
         */
        DenseScoreVector(uint32 numElements);

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

        const AbstractEvaluatedPrediction* processScores(const AbstractEvaluatedPrediction* bestHead,
                                                         IScoreProcessor& scoreProcessor) const override;

};
