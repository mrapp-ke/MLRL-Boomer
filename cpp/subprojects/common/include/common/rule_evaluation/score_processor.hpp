/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/rule_evaluation/score_vector_dense.hpp"
#include "common/rule_evaluation/score_vector_binned_dense.hpp"
#include "common/indices/index_vector_complete.hpp"
#include "common/indices/index_vector_partial.hpp"
#include "common/head_refinement/prediction_evaluated.hpp"
#include "common/statistics/statistics_subset.hpp"


/**
 * Allows to process the scores that are stored by an `IScoreVector` in order to convert them into the head of a rule,
 * represented by an `AbstractEvaluatedPrediction`.
 */
class ScoreProcessor {

    private:

        std::unique_ptr<AbstractEvaluatedPrediction> headPtr_;

    public:

        /**
         * Processes the scores that are stored by a `DenseScoreVector<CompleteIndexVector>` in order to convert them
         * into the head of a rule.
         *
         * @param bestHead      A pointer to an object of type `AbstractEvaluatedPrediction` that represents the best
         *                      head that has been created so far
         * @param scoreVector   A reference to an object of type `DenseScoreVector<CompleteIndexVector>` that stores the
         *                      scores to be processed
         * @return              A pointer to an object of type `AbstractEvaluatedPrediction` that has been created or a
         *                      null pointer if no object has been created
         */
        const AbstractEvaluatedPrediction* processScores(const AbstractEvaluatedPrediction* bestHead,
                                                         const DenseScoreVector<CompleteIndexVector>& scoreVector);

        /**
         * Processes the scores that are stored by a `DenseScoreVector<PartialIndexVector>` in order to convert them
         * into the head of a rule.
         *
         * @param bestHead      A pointer to an object of type `AbstractEvaluatedPrediction` that represents the best
         *                      head that has been created so far
         * @param scoreVector   A reference to an object of type `DenseScoreVector<PartialIndexVector>` that stores the
         *                      scores to be processed
         * @return              A pointer to an object of type `AbstractEvaluatedPrediction` that has been created or a
         *                      null pointer if no object has been created
         */
        const AbstractEvaluatedPrediction* processScores(const AbstractEvaluatedPrediction* bestHead,
                                                         const DenseScoreVector<PartialIndexVector>& scoreVector);

        /**
         * Processes the scores that are stored by a `DenseBinnedScoreVector<CompleteIndexVector>` in order to convert
         * them into the head of a rule.
         *
         * @param bestHead      A pointer to an object of type `AbstractEvaluatedPrediction` that represents the best
         *                      head that has been created so far
         * @param scoreVector   A reference to an object of type `DenseBinnedScoreVector<CompleteIndexVector>` that
         *                      stores the scores to be processed
         * @return              A pointer to an object of type `AbstractEvaluatedPrediction` that has been created or a
         *                      null pointer if no object has been created
         */
        const AbstractEvaluatedPrediction* processScores(
            const AbstractEvaluatedPrediction* bestHead,
            const DenseBinnedScoreVector<CompleteIndexVector>& scoreVector);

        /**
         * Processes the scores that are stored by a `DenseBinnedScoreVector<PartialIndexVector>` in order to convert
         * them into the head of a rule.
         *
         * @param bestHead      A pointer to an object of type `AbstractEvaluatedPrediction` that represents the best
         *                      head that has been created so far
         * @param scoreVector   A reference to an object of type `DenseBinnedScoreVector<PartialIndexVector>` that
         *                      stores the scores to be processed
         * @return              A pointer to an object of type `AbstractEvaluatedPrediction` that has been created or a
         *                      null pointer if no object has been created
         */
        const AbstractEvaluatedPrediction* processScores(const AbstractEvaluatedPrediction* bestHead,
                                                         const DenseBinnedScoreVector<PartialIndexVector>& scoreVector);

        /**
         * TODO
         *
         * @param bestHead          TODO
         * @param statisticsSubset  TODO
         * @param uncovered         TODO
         * @param accumulated       TODO
         * @return                  TODO
         */
        const AbstractEvaluatedPrediction* findHead(const AbstractEvaluatedPrediction* bestHead,
                                                    IStatisticsSubset& statisticsSubset, bool uncovered,
                                                    bool accumulated);

        /**
         * TODO
         *
         * @return TODO
         */
        std::unique_ptr<AbstractEvaluatedPrediction> pollHead();

};
