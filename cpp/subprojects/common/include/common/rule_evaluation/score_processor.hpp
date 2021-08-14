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
         * Finds the best head for a rule, given the predictions that are provided by a `IStatisticsSubset`.
         *
         * The given object of type `IStatisticsSubset` must have been prepared properly via calls to the function
         * `IStatisticsSubset#addToSubset`.
         *
         * @param bestHead          A pointer to an object of type `AbstractEvaluatedPrediction` that corresponds to the
         *                          best rule known so far (as found in the previous or current refinement iteration) or
         *                          a null pointer, if no such rule is available yet. The new head must be better than
         *                          this one, otherwise it is discarded
         * @param statisticsSubset  A reference to an object of type `IStatisticsSubset` to be used for calculating
         *                          predictions and corresponding quality scores
         * @param uncovered         False, if the rule for which the head should be found covers all statistics that
         *                          have been added to the `IStatisticsSubset` so far, True, if the rule covers all
         *                          statistics that have not been added yet
         * @param accumulated       False, if the rule covers all statistics that have been added since the
         *                          `IStatisticsSubset` has been reset for the last time, True, if the rule covers all
         *                          statistics that have been added so far
         * @return                  A pointer to an object of type `AbstractEvaluatedPrediction`, representing the head
         *                          that has been found or a null pointer if the head that has been found is not better
         *                          than `bestHead`
         */
        const AbstractEvaluatedPrediction* findHead(const AbstractEvaluatedPrediction* bestHead,
                                                    IStatisticsSubset& statisticsSubset, bool uncovered,
                                                    bool accumulated);

        /**
         * Returns the best head that has been found by the function `findHead.
         *
         * @return An unique pointer to an object of type `AbstractEvaluatedPrediction`, representing the best head that
         *         has been found
         */
        std::unique_ptr<AbstractEvaluatedPrediction> pollHead();

};
