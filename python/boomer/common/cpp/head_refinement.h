/**
 * Provides classes that implement different strategies for finding the heads of rules.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "statistics.h"


/**
 * Defines an interface for all classes that allow to find the best head for a rule.
 */
class IHeadRefinement {

    public:

        virtual ~IHeadRefinement() { };

        /**
         * Finds the best head for a rule, given the predictions that are provided by a `IStatisticsSubset`.
         *
         * The given object of type `IStatisticsSubset` must have been prepared properly via calls to the function
         * `IStatisticsSubset#addToSubset`.
         *
         * @param bestHead          A pointer to an object of type `PredictionCandidate` that corresponds to the best
         *                          rule known so far (as found in the previous or current refinement iteration) or a
         *                          null pointer, if no such rule is available yet. The new head must be better than
         *                          this one, otherwise it is discarded
         * @param headPtr           An unique pointer to an object of type `PredictionCandidate`, which represents the
         *                          best head that has been found so far. If the pointer does not refer to an object, a
         *                          new object will be created, otherwise the existing object will be modified to avoid
         *                          unnecessary memory allocations
         * @param labelIndices      A pointer to an array of type `uint32`, shape `(num_predictions)`, representing the
         *                          indices of the labels for which the head may predict or a null pointer, if the head
         *                          may predict for all labels
         * @param statisticsSubset  A reference to an object of type `IStatisticsSubset` to be used for calculating
         *                          predictions and corresponding quality scores
         * @param uncovered         False, if the rule for which the head should be found covers all statistics that
         *                          have been added to the `IStatisticsSubset` so far, True, if the rule covers all
         *                          statistics that have not been added yet
         * @param accumulated       False, if the rule covers all statistics that have been added since the
         *                          `IStatisticsSubset` has been reset for the last time, True, if the rule covers all
         *                          statistics that have been added so far
         * @return                  True, if the head that has been found is better than `bestHead`, false otherwise
         */
        virtual bool findHead(const PredictionCandidate* bestHead, std::unique_ptr<PredictionCandidate>& headPtr,
                              const uint32* labelIndices, IStatisticsSubset& statisticsSubset, bool uncovered,
                              bool accumulated) const = 0;

        /**
         * Returns the best head that has been found by the function `findHead`.
         *
         * @return An unique pointer to an object of type `PredictionCandidate` that stores the scores that are
         *         predicted by the best head that has been found
         */
        virtual std::unique_ptr<PredictionCandidate> pollHead() = 0;

        /**
         * Calculates the optimal scores to be predicted by a rule, as well as the rule's overall quality score,
         * according to a `IStatisticsSubset`.
         *
         * The given object of type `IStatisticsSubset` must have been prepared properly via calls to the function
         * `IStatisticsSubset#addToSubset`.
         *
         * @param statisticsSubset  A reference to an object of type `IStatisticsSubset` to be used for calculating
         *                          predictions and corresponding quality scores
         * @param uncovered         False, if the rule for which the optimal scores should be calculated covers all
         *                          statistics that have been added to the `IStatisticsSubset` so far, True, if the rule
         *                          covers all statistics that have not been added yet
         * @param accumulated       False, if the rule covers all examples that have been added since the
         *                          `IStatisticsSubset` has been reset for the last time, True, if the rule covers all
         *                          examples that have been added so far
         * @return                  A reference to an object of type `EvaluatedPrediction` that stores the optimal
         *                          scores to be predicted by the rule, as well as its overall quality score
         */
        virtual const EvaluatedPrediction& calculatePrediction(IStatisticsSubset& statisticsSubset, bool uncovered,
                                                               bool accumulated) const = 0;

};

/**
 * Defines an interface for all factories that allow to create instances of the type `IHeadRefinement`.
 */
class IHeadRefinementFactory {

    public:

        virtual ~IHeadRefinementFactory() { };

        /**
         * Creates and returns a new object of type `IHeadRefinement`.
         *
         * @return An unique pointer to an object of type `IHeadRefinement` that has been created
         */
        virtual std::unique_ptr<IHeadRefinement> create() const = 0;

};

/**
 * Allows to find the best single-label head that predicts for a single label.
 */
class SingleLabelHeadRefinementImpl : virtual public IHeadRefinement {

    private:

        std::unique_ptr<PredictionCandidate> headPtr_;

    public:

        bool findHead(const PredictionCandidate* bestHead, std::unique_ptr<PredictionCandidate>& headPtr,
                      const uint32* labelIndices, IStatisticsSubset& statisticsSubset, bool uncovered,
                      bool accumulated) const override;

        std::unique_ptr<PredictionCandidate> pollHead() override;

        const EvaluatedPrediction& calculatePrediction(IStatisticsSubset& statisticsSubset, bool uncovered,
                                                       bool accumulated) const override;

};

/**
 * Allows to create instances of the class `SingleLabelHeadRefinementImpl`.
 */
class SingleLabelHeadRefinementFactoryImpl : virtual public IHeadRefinementFactory {

    public:

        std::unique_ptr<IHeadRefinement> create() const override;

};

/**
 * Allows to find the best multi-label head that predicts for all labels.
 */
class FullHeadRefinementImpl : virtual public IHeadRefinement {

    private:

        std::unique_ptr<PredictionCandidate> headPtr_;

    public:

        bool findHead(const PredictionCandidate* bestHead, std::unique_ptr<PredictionCandidate>& headPtr,
                      const uint32* labelIndices, IStatisticsSubset& statisticsSubset, bool uncovered,
                      bool accumulated) const override;

        std::unique_ptr<PredictionCandidate> pollHead() override;

        const EvaluatedPrediction& calculatePrediction(IStatisticsSubset& statisticsSubset, bool uncovered,
                                                       bool accumulated) const override;

};

/**
 * Allows to create instances of the class `FullHeadRefinementImpl`.
 */
class FullHeadRefinementFactoryImpl : virtual public IHeadRefinementFactory {

    public:

        std::unique_ptr<IHeadRefinement> create() const override;

};
