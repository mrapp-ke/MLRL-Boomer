/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/rule_evaluation/score_vector_label_wise_dense.hpp"
#include "common/rule_evaluation/score_vector_label_wise_binned_dense.hpp"
#include "common/indices/index_vector_complete.hpp"
#include "common/indices/index_vector_partial.hpp"

// Forward declarations
class AbstractEvaluatedPrediction;


/**
 * Defines an interface for all classes that process the scores that are stored by an `ILabelWiseScoreVector` in order
 * to convert them into the head of a rule, represented by an `AbstractEvaluatedPrediction`.
 */
class ILabelWiseScoreProcessor {

    public:

        virtual ~ILabelWiseScoreProcessor() { };

        /**
         * Processes the scores that are stored by a `DenseLabelWiseScoreVector<CompleteIndexVector>` in order to
         * convert them into the head of a rule.
         *
         * @param bestHead      A pointer to an object of type `AbstractEvaluatedPrediction` that represents the best
         *                      head that has been created so far
         * @param scoreVector   A reference to an object of type `DenseLabelWiseScoreVector<CompleteIndexVector>` that
         *                      stores the scores to be processed
         * @return              A pointer to an object of type `AbstractEvaluatedPrediction` that has been created or a
         *                      null pointer if no object has been created
         */
        virtual const AbstractEvaluatedPrediction* processScores(
            const AbstractEvaluatedPrediction* bestHead,
            const DenseLabelWiseScoreVector<CompleteIndexVector>& scoreVector) = 0;

        /**
         * Processes the scores that are stored by a `DenseLabelWiseScoreVector<PartialIndexVector>` in order to convert
         * them into the head of a rule.
         *
         * @param bestHead      A pointer to an object of type `AbstractEvaluatedPrediction` that represents the best
         *                      head that has been created so far
         * @param scoreVector   A reference to an object of type `DenseLabelWiseScoreVector<PartialIndexVector>` that
         *                      stores the scores to be processed
         * @return              A pointer to an object of type `AbstractEvaluatedPrediction` that has been created or a
         *                      null pointer if no object has been created
         */
        virtual const AbstractEvaluatedPrediction* processScores(
            const AbstractEvaluatedPrediction* bestHead,
            const DenseLabelWiseScoreVector<PartialIndexVector>& scoreVector) = 0;

        /**
         * Processes the scores that are stored by a `DenseBinnedLabelWiseScoreVector<CompleteIndexVector>` in order to
         * convert them into the head of a rule.
         *
         * @param bestHead      A pointer to an object of type `AbstractEvaluatedPrediction` that represents the best
         *                      head that has been created so far
         * @param scoreVector   A reference to an object of type `DenseBinnedLabelWiseScoreVector<CompleteIndexVector>`
         *                      that stores the scores to be processed
         * @return              A pointer to an object of type `AbstractEvaluatedPrediction` that has been created or a
         *                      null pointer if no object has been created
         */
        virtual const AbstractEvaluatedPrediction* processScores(
            const AbstractEvaluatedPrediction* bestHead,
            const DenseBinnedLabelWiseScoreVector<CompleteIndexVector>& scoreVector) = 0;

        /**
         * Processes the scores that are stored by a `DenseBinnedLabelWiseScoreVector<PartialIndexVector>` in order to
         * convert them into the head of a rule.
         *
         * @param bestHead      A pointer to an object of type `AbstractEvaluatedPrediction` that represents the best
         *                      head that has been created so far
         * @param scoreVector   A reference to an object of type `DenseBinnedLabelWiseScoreVector<PartialIndexVector>`
         *                      that stores the scores to be processed
         * @return              A pointer to an object of type `AbstractEvaluatedPrediction` that has been created or a
         *                      null pointer if no object has been created
         */
        virtual const AbstractEvaluatedPrediction* processScores(
            const AbstractEvaluatedPrediction* bestHead,
            const DenseBinnedLabelWiseScoreVector<PartialIndexVector>& scoreVector) = 0;

};
