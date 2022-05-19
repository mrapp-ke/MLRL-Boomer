/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 */
#pragma once

#include "common/statistics/statistics_weighted.hpp"
#include "common/rule_refinement/prediction_complete.hpp"
#include "common/rule_refinement/prediction_partial.hpp"


/**
 * Defines an interface for all classes that provide access to statistics about the labels of the training examples,
 * which serve as the basis for learning a new rule or refining an existing one.
 */
class IStatistics {

    public:

        virtual ~IStatistics() { };

        /**
         * Returns the number of available statistics.
         *
         * @return The number of statistics
         */
        virtual uint32 getNumStatistics() const = 0;

        /**
         * Returns the number of available labels.
         *
         * @return The number of labels
         */
        virtual uint32 getNumLabels() const = 0;

        /**
         * Updates a specific statistic based on the prediction of a rule that predicts for all available labels.
         *
         * This function must be called for each statistic that is covered by the new rule before learning the next
         * rule.
         *
         * @param statisticIndex    The index of the statistic to be updated
         * @param prediction        A reference to an object of type `CompletePrediction` that stores the scores that
         *                          are predicted by the rule
         */
        virtual void applyPrediction(uint32 statisticIndex, const CompletePrediction& prediction) = 0;

        /**
         * Updates a specific statistic based on the prediction of a rule that predicts for a subset of the available
         * labels.
         *
         * This function must be called for each statistic that is covered by the new rule before learning the next
         * rule.
         *
         * @param statisticIndex    The index of the statistic to be updated
         * @param prediction        A reference to an object of type `PartialPrediction` that stores the scores that are
         *                          predicted by the rule
         */
        virtual void applyPrediction(uint32 statisticIndex, const PartialPrediction& prediction) = 0;

        /**
         * Calculates and returns a numeric score that assesses the quality of the current predictions for a specific
         * statistic.
         *
         * @param statisticIndex    The index of the statistic for which the predictions should be evaluated
         * @return                  The numeric score that has been calculated
         */
        virtual float64 evaluatePrediction(uint32 statisticIndex) const = 0;

        /**
         * Creates and returns a new object of type `IWeightedStatistics`.
         *
         * @return An unique pointer to an object of type `IWeightedStatistics` that has been created
         */
        virtual std::unique_ptr<IWeightedStatistics> createWeightedStatistics() const = 0;

};
