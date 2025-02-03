/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"
#include "mlrl/common/rule_evaluation/score_vector_binned_dense.hpp"
#include "mlrl/common/rule_evaluation/score_vector_dense.hpp"
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
         * Processes the scores that are stored by a `DenseScoreVector`, which stores 32-bit floating point values for
         * all available outputs, in order to convert them into the head of a rule.
         *
         * @param scoreVector A reference to an object of type `DenseScoreVector` that stores the scores to be processed
         */
        void processScores(const DenseScoreVector<float32, CompleteIndexVector>& scoreVector);

        /**
         * Processes the scores that are stored by a `DenseScoreVector`, which stores 64-bit floating point values for
         * all available outputs, in order to convert them into the head of a rule.
         *
         * @param scoreVector A reference to an object of type `DenseScoreVector` that stores the scores to be processed
         */
        void processScores(const DenseScoreVector<float64, CompleteIndexVector>& scoreVector);

        /**
         * Processes the scores that are stored by a `DenseScoreVector`, which stores 32-bit floating point values for
         * a subset of the available outputs, in order to convert them into the head of a rule.
         *
         * @param scoreVector A reference to an object of type `DenseScoreVector` that stores the scores to be processed
         */
        void processScores(const DenseScoreVector<float32, PartialIndexVector>& scoreVector);

        /**
         * Processes the scores that are stored by a `DenseScoreVector`, which stores 64-bit floating point values for
         * a subset of the available outputs, in order to convert them into the head of a rule.
         *
         * @param scoreVector A reference to an object of type `DenseScoreVector` that stores the scores to be processed
         */
        void processScores(const DenseScoreVector<float64, PartialIndexVector>& scoreVector);

        /**
         * Processes the scores that are stored by a `DenseBinnedScoreVector`, which stores 32-bit floating point values
         * for all available outputs, in order to convert them into the head of a rule.
         *
         * @param scoreVector A reference to an object of type `DenseScoreVector` that stores the scores to be processed
         */
        void processScores(const DenseBinnedScoreVector<float32, CompleteIndexVector>& scoreVector);

        /**
         * Processes the scores that are stored by a `DenseBinnedScoreVector`, which stores 64-bit floating point values
         * for all available outputs, in order to convert them into the head of a rule.
         *
         * @param scoreVector A reference to an object of type `DenseScoreVector` that stores the scores to be processed
         */
        void processScores(const DenseBinnedScoreVector<float64, CompleteIndexVector>& scoreVector);

        /**
         * Processes the scores that are stored by a `DenseBinnedScoreVector`, which stores 32-bit floating point values
         * for a subset of the available available outputs, in order to convert them into the head of a rule.
         *
         * @param scoreVector A reference to an object of type `DenseBinnedScoreVector` that stores the scores to be
         *                    processed
         */
        void processScores(const DenseBinnedScoreVector<float32, PartialIndexVector>& scoreVector);

        /**
         * Processes the scores that are stored by a `DenseBinnedScoreVector`, which stores 64-bit floating point values
         * for a subset of the available available outputs, in order to convert them into the head of a rule.
         *
         * @param scoreVector A reference to an object of type `DenseBinnedScoreVector` that stores the scores to be
         *                    processed
         */
        void processScores(const DenseBinnedScoreVector<float64, PartialIndexVector>& scoreVector);

        /**
         * Processes the scores that are stored by a `IScoreVector` in order to convert them into the head of a rule.
         *
         * @param scoreVector A reference to an object of type `DenseBinnedScoreVector` that stores the scores to be
         *                    processed
         */
        void processScores(const IScoreVector& scoreVector);
};
