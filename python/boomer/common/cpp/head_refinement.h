/**
 * Provides classes that implement different strategies for finding the heads of rules.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"
#include "predictions.h"
#include "statistics.h"


/**
 * An abstract base class for all classes that allow to find the best head for a rule.
 */
class AbstractHeadRefinement {

    public:

        virtual ~AbstractHeadRefinement();

        /**
         * Finds and returns the best head for a rule given the predictions that are provided by a
         * `AbstractRefinementSearch`.
         *
         * The given object of type `AbstractRefinementSearch` must have been prepared properly via calls to the
         * function `AbstractRefinementSearch#updateSearch`.
         *
         * @param bestHead          A pointer to an object of type `PredictionCandidate` that corresponds to the best
         *                          rule known so far (as found in the previous or current refinement iteration) or
         *                          NULL, if no such rule is available yet. The new head must be better than this one,
         *                          otherwise it is discarded
         * @param recyclableHead    A pointer to an object of type `PredictionCandidate` that may be modified instead of
         *                          creating a new instance to avoid unnecessary memory allocations or NULL, if no such
         *                          object is available
         * @param labelIndices      A pointer to an array of type `uint32`, shape `(num_predictions)`, representing the
         *                          indices of the labels for which the head may predict or NULL, if the head may
         *                          predict for all labels
         * @param refinementSearch  A pointer to an object of type `AbstractRefinementSearch` to be used for calculating
         *                          predictions and corresponding quality scores
         * @param uncovered         False, if the rule for which the head should be found covers all examples that have
         *                          been provided to the `AbstractRefinementSearch` so far, True, if the rule covers all
         *                          examples that have not been provided yet
         * @param accumulated       False, if the rule covers all examples that have been provided since the
         *                          `AbstractRefinementSearch` has been reset for the last time, True, if the rule
         *                          covers all examples that have been provided so far
         * @return                  A pointer to an object of type 'PredictionCandidate' that stores information about
         *                          the head that has been found, if the head is better than `bestHead`, NULL otherwise
         */
        virtual PredictionCandidate* findHead(PredictionCandidate* bestHead, PredictionCandidate* recyclableHead,
                                              const uint32* labelIndices, AbstractRefinementSearch* refinementSearch,
                                              bool uncovered, bool accumulated);

        /**
         * Calculates the optimal scores to be predicted by a rule, as well as the rule's overall quality score,
         * according to a `AbstractRefinementSearch`.
         *
         * The given object of type `AbstractRefinementSearch` must have been prepared properly via calls to the
         * function `AbstractRefinementSearch#updateSearch`.
         *
         * @param refinementSearch  A pointer to an object of type `AbstractRefinementSearch` to be used for calculating
         *                          predictions and corresponding quality scores
         * @param uncovered         False, if the rule for which the optimal scores should be calculated covers all
         *                          examples that have been provided to the `RefinementSearch` so far, True, if the rule
         *                          covers all examples that have not been provided yet
         * @param accumulated       False, if the rule covers all examples that have been provided since the
         *                          `RefinementSearch` has been reset for the last time, True, if the rule covers all
         *                          examples that have been  provided so far
         * @return                  A pointer to an object of type `PredictionCandidate` that stores the optimal scores
         *                          to be predicted by the rule, as well as its overall quality score
         */
        virtual PredictionCandidate* calculatePrediction(AbstractRefinementSearch* refinementSearch, bool uncovered,
                                                         bool accumulated);

};

/**
 * Allows to find the best single-label head that predicts for a single label.
 */
class SingleLabelHeadRefinementImpl : public AbstractHeadRefinement {

    public:

        ~SingleLabelHeadRefinementImpl();

        PredictionCandidate* findHead(PredictionCandidate* bestHead, PredictionCandidate* recyclableHead,
                                      const uint32* labelIndices, AbstractRefinementSearch* refinementSearch,
                                      bool uncovered, bool accumulated) override;

        PredictionCandidate* calculatePrediction(AbstractRefinementSearch* refinementSearch, bool uncovered,
                                                 bool accumulated) override;

};

/**
 * Allows to find the best multi-label head that predicts for all labels.
 */
class FullHeadRefinementImpl : public AbstractHeadRefinement {

    public:

        ~FullHeadRefinementImpl();

        PredictionCandidate* findHead(PredictionCandidate* bestHead, PredictionCandidate* recyclableHead,
                                      const uint32* labelIndices, AbstractRefinementSearch* refinementSearch,
                                      bool uncovered, bool accumulated) override;

        PredictionCandidate* calculatePrediction(AbstractRefinementSearch* refinementSearch, bool uncovered,
                                                 bool accumulated) override;

};
