/**
 * Implements classes that provide access to statistics about the labels of training examples.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"
#include "predictions.h"
#include "data.h"
#include "binning.h"
#include <memory>


/**
 * Defines an interface for all classes that provide access to a subset of the statistics that are stored by an instance
 * of the class `AbstractStatistics` and allows to calculate the scores to be predicted by rules that cover such a
 * subset.
 */
class IStatisticsSubset {

    public:

        virtual ~IStatisticsSubset() { };

        /**
         * Adds the statistics at a specific index to the subset in order to mark it as covered by the condition that is
         * currently considered for refining a rule.
         *
         * This function must be called repeatedly for each statistic that is covered by the current condition,
         * immediately after the invocation of the function `Statistics#createSubset`. Each of these statistics must
         * have been provided earlier via the function `Statistics#addSampledStatistic` or
         * `Statistics#updateCoveredStatistic`.
         *
         * This function is supposed to update any internal state of the subset that relates to the statistics that are
         * covered by the current condition, i.e., to compute and store local information that is required by the other
         * functions that will be called later. Any information computed by this function is expected to be reset when
         * invoking the function `resetSubset` for the next time.
         *
         * @param statisticIndex    The index of the covered statistic
         * @param weight            The weight of the covered statistic
         */
        virtual void addToSubset(uint32 statisticIndex, uint32 weight) = 0;

        /**
         * Resets the subset by removing all statistics that have been added via preceding calls to the function
         * `addToSubset`.
         *
         * This function is supposed to reset the internal state of the subset to the state when the subset was created
         * via the function `Statistics#createSubset`. When calling this function, the current state must not be purged
         * entirely, but it must be cached and made available for use by the functions `calculateExampleWisePrediction`
         * and `calculateLabelWisePrediction` (if the function argument `accumulated` is set accordingly).
         *
         * This function may be invoked multiple times with one or several calls to `addToSubset` in between, which is
         * supposed to update the previously cached state by accumulating the current one, i.e., the accumulated cached
         * state should be the same as if `resetSubset` would not have been called at all.
         */
        virtual void resetSubset() = 0;

        /**
         * Calculates and returns the scores to be predicted by a rule that covers all statistics that have been added
         * to the subset so far via the function `addToSubset`.
         *
         * If the argument `uncovered` is 1, the rule is considered to cover all statistics that belong to the
         * difference between the statistics that have been provided via the function `Statistics#addSampledStatistic`
         * or `Statistics#updateCoveredStatistic` and the statistics that have been added to the subset via the function
         * `addToSubset`.
         *
         * If the argument `accumulated` is 1, all statistics that have been added since the subset has been created
         * via the function `Statistics#createSubset` are taken into account even if the function `resetSubset` has been
         * called since then. If said function has not been invoked, this argument does not have any effect.
         *
         * The calculated scores correspond to the subset of labels that have been provided when creating the subset via
         * the function `Statistics#createSubset`. The score to be predicted for an individual label is calculated
         * independently of the other labels, i.e., in the non-decomposable case it is assumed that the rule will not
         * predict for any other labels. In addition to each score, a quality score, which assesses the quality of the
         * prediction for the respective label, is returned.
         *
         * @param uncovered     0, if the rule covers all statistics that have been added to the subset via the function
         *                      `addToSubset`, 1, if the rule covers all statistics that belong to the difference
         *                      between the statistics that have been provided via the function
         *                      `Statistics#addSampledStatistic` or `Statistics#updateCoveredStatistic` and the
         *                      statistics that have been added via the function `addToSubset`
         * @param accumulated   0, if the rule covers all statistics that have been added to the subset via the function
         *                      `addToSubset` since the function `resetSubset` has been called for the last time, 1, if
         *                      the rule covers all examples that have been provided since the subset has been created
         *                      via the function `Statistics#createSubset`
         * @return              A reference to an object of type `LabelWisePredictionCandidate` that stores the scores
         *                      to be predicted by the rule for each considered label, as well as the corresponding
         *                      quality scores
         */
        virtual const LabelWisePredictionCandidate& calculateLabelWisePrediction(bool uncovered, bool accumulated) = 0;

        /**
         * Calculates and returns the scores to be predicted by a rule that covers all statistics that have been added
         * to the subset so far via the function `addToSubset`.
         *
         * If the argument `uncovered` is 1, the rule is considered to cover all statistics that belong to the
         * difference between the statistics that have been provided via the function `Statistics#addSampledStatistic`
         * or `Statistics#updateCoveredStatistic` and the statistics that have been added to the subset via the function
         * `addToSubset`.
         *
         * If the argument `accumulated` is 1, all statistics that have been added since the subset has been created
         * via the function `Statistics#createSubset` are taken into account even if the function `resetSubset` has been
         * called since then. If said function has not been invoked, this argument does not have any effect.
         *
         * The calculated scores correspond to the subset of labels that have been provided when creating the subset via
         * the function `Statistics#createSubset`. The score to be predicted for an individual label is calculated with
         * respect to the predictions for the other labels. In the decomposable case, i.e., if the labels are considered
         * independently of each other, this function is equivalent to the function `calculateLabelWisePrediction`. In
         * addition to the scores, an overall quality score, which assesses the quality of the predictions for all
         * labels in terms of a single score, is returned.
         *
         * @param uncovered     0, if the rule covers all statistics that have been added to the subset via the function
         *                      `addToSubset`, 1, if the rule covers all statistics that belong to the difference
         *                      between the statistics that have been provided via the function
         *                      `Statistics#addSampledStatistic` or `Statistics#updateCoveredStatistic` and the
         *                      statistics that have been added via the function `addToSubset`
         * @param accumulated   0, if the rule covers all statistics that have been added to the subset via the function
         *                      `addToSubset` since the function `resetSubset` has been called for the last time, 1, if
         *                      the rule covers all examples that have been provided since the subset has been created
         *                      via the function `Statistics#createSubset`
         * @return              A reference to an object of type `PredictionCandidate` that stores the scores to be
         *                      predicted by the rule for each considered label, as well as an overall quality score
         */
        virtual const PredictionCandidate& calculateExampleWisePrediction(bool uncovered, bool accumulated) = 0;

};

/**
 * An abstract base class for all classes that provide access to a subset of the statistics that are stores by an
 * instance of the class `AbstractStatistics` and allow to calculate the scores to be predicted by rules that cover such
 * a subset in the decomposable case, i.e., if the label-wise predictions are the same as the example-wise predictions.
 */
class AbstractDecomposableStatisticsSubset : virtual public IStatisticsSubset {

    public:

        const PredictionCandidate& calculateExampleWisePrediction(bool uncovered, bool accumulated) override;

};

/**
 * An abstract base class for all classes that provide access to statistics about the labels of the training examples,
 * which serve as the basis for learning a new rule or refining an existing one.
 */
class AbstractStatistics : virtual public IMatrix {

    private:

        uint32 numStatistics_;

        uint32 numLabels_;

    public:

        /**
         * Defines an interface for all classes that allow to build histograms by aggregating the statistics that
         * correspond to the same bins.
         */
        class IHistogramBuilder : virtual public IBinningObserver {

            public:

                virtual ~IHistogramBuilder() { };

                /**
                 * Creates and returns a new instance of the class `AbstractStatistics` that stores the histogram that
                 * has been built.
                 *
                 * @return An unique pointer to an object of type `AbstractStatistics` that has been created
                 */
                virtual std::unique_ptr<AbstractStatistics> build() const = 0;

        };

        /**
         * @param numStatistics The number of statistics
         */
        AbstractStatistics(uint32 numStatistics, uint32 numLabels);

        /**
         * Resets the statistics which should be considered in the following for learning a new rule. The indices of the
         * respective statistics must be provided via subsequent calls to the function `addSampledStatistic`.
         *
         * This function must be invoked before a new rule is learned from scratch, as each rule may be learned on a
         * different sub-sample of the statistics.
         *
         * This function is supposed to reset any non-global internal state that only holds for a certain subset of the
         * available statistics and therefore becomes invalid when a different subset of the statistics should be used.
         */
        virtual void resetSampledStatistics() = 0;

        /**
         * Adds a specific statistic to the sub-sample that should be considered in the following for learning a new
         * rule from scratch.
         *
         * This function must be called repeatedly for each statistic that should be considered, immediately after the
         * invocation of the function `resetSampledStatistics`.
         *
         * This function is supposed to update any internal state that relates to the considered statistics, i.e., to
         * compute and store local information that is required by the other function that will be called later. Any
         * information computed by this function is expected to be reset when invoking the function
         * `resetSampledStatistics` for the next time.
         *
         * @param statisticIndex    The index of the statistic that should be considered
         * @param weight            The weight of the statistic that should be considered
         */
        virtual void addSampledStatistic(uint32 statisticIndex, uint32 weight) = 0;

        /**
         * Resets the statistics which should be considered in the following for refining an existing rule. The indices
         * of the respective statistics must be provided via subsequent calls to the function `updateCoveredStatistic`.
         *
         * This function must be invoked each time an existing rule has been refined, i.e., when a new condition has
         * been added to its body, because this results in a subset of the statistics being covered by the refined rule.
         *
         * This function is supposed to reset any non-global internal state that only holds for a certain subset of the
         * available statistics and therefore becomes invalid when a different subset of the statistics should be used.
         */
        virtual void resetCoveredStatistics() = 0;

        /**
         * Adds a specific statistic to the subset that is covered by an existing rule and therefore should be
         * considered in the following for refining an existing rule.
         *
         * This function must be called repeatedly for each statistic that is covered by the existing rule, immediately
         * after the invocation of the function `resetCoveredStatistics`.
         *
         * Alternatively, this function may be used to indicate that a statistic, which has previously been passed to
         * this function, should not be considered anymore by setting the argument `remove` accordingly.
         *
         * This function is supposed to update any internal state that relates to the considered statistics, i.e., to
         * compute and store local information that is required by the other function that will be called later. Any
         * information computed by this function is expected to be reset when invoking the function
         * `resetCoveredStatistics` for the next time.
         *
         * @param statisticIndex    The index of the statistic that should be updated
         * @param weight            The weight of the statistic that should be updated
         * @param remove            False, if the statistic should be considered, True, if the statistic should not be
         *                          considered anymore
         */
        virtual void updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove) = 0;

        /**
         * Creates a new, empty subset of the statistics. Individual statistics that are covered by a refinement of a
         * rule can be added to the subset via subsequent calls to the function `IStatisticsSubset#addToSubset`.
         *
         * This function must be called each time a new refinement is considered, unless the refinement covers all
         * statistics previously provided via calls to the function `IStatisticsSubset#addToSubset`.
         *
         * Optionally, a subset of the available labels may be specified via the argument `labelIndices`. In such case,
         * only the statistics that correspond to the specified labels will be included in the subset. When calling this
         * function again to create a new subset from scratch, a different set of labels may be specified.
         *
         * @param numLabelIndices   The number of elements in the array `labelIndices`
         * @param labelIndices      A pointer to an array of type `uint32`, shape `(numPredictions)`, representing the
         *                          indices of the labels that should be included in the subset or None, if all labels
         *                          should be included
         * @return                  An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(uint32 numLabelIndices,
                                                                const uint32* labelIndices) const = 0;

        /**
         * Updates a specific statistic based on the predictions of a newly induced rule.
         *
         * This function must be called for each statistic that is covered by the new rule before learning the next
         * rule.
         *
         * @param statisticIndex    The index of the statistic to be updated
         * @param prediction        A reference to an object of type `Prediction`, representing the predictions of the
         *                          newly induced rule
         */
        virtual void applyPrediction(uint32 statisticIndex, const Prediction& prediction) = 0;

        /**
         * Creates and returns a new instance of the class `IHistogramBuilder` that allows to build a histogram based on
         * the statistics.
         *
         * @return An unique pointer to an object of type `IHistogramBuilder` that has been created
         */
        virtual std::unique_ptr<IHistogramBuilder> buildHistogram(uint32 numBins) const = 0;

        uint32 getNumRows() const override;

        uint32 getNumCols() const override;

};
