/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/rule_evaluation/score_vector.hpp"
#include "mlrl/common/rule_refinement/prediction_evaluated.hpp"

#include <memory>

/**
 * Allows to process the scores that are stored by an `IScoreVector` in order to convert them into the head of a rule,
 * represented by an `IEvaluatedPrediction`.
 */
class ScoreProcessor {
    private:

        std::unique_ptr<IEvaluatedPrediction>& headPtr_;

    public:

        /**
         * @param headPtr   A reference to an unique pointer of type `IEvaluatedPrediction` that should be used to store
         *                  the rule head that is created by the processor
         */
        explicit ScoreProcessor(std::unique_ptr<IEvaluatedPrediction>& headPtr);

        /**
         * Processes the scores that are stored by a `IScoreVector` in order to convert them into the head of a rule.
         *
         * @param scoreVector A reference to an object of type `IScoreVector` that stores the scores to be processed
         */
        void processScores(const IScoreVector& scoreVector);
};
