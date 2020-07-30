/**
 * Implements classes that provide access to the labels of training examples.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"
#include "sparse.h"
#include "rule_evaluation.h"


namespace statistics {

    /**
     * An abstract base class for all label matrices that provide random access to the labels of the training examples.
     */
    class AbstractLabelMatrix {

        public:

            /**
             * @param numExamples   The number of examples
             * @param numLabels     The number of labels
             */
            AbstractLabelMatrix(intp numExamples, intp numLabels);

            virtual ~AbstractLabelMatrix();

            /**
             * The number of examples.
             */
            intp numExamples_;

            /**
             * The number of labels.
             */
            intp numLabels_;

            /**
             * Returns whether a specific label of the example at a given index is relevant or irrelevant.
             *
             * @param exampleIndex  The index of the example
             * @param labelIndex    The index of the label
             * @return              1, if the label is relevant, 0 otherwise
             */
            virtual uint8 getLabel(intp exampleIndex, intp labelIndex);

    };

    /**
     * Implements random access to the labels of the training examples based on a C-contiguous array.
     */
    class DenseLabelMatrixImpl : public AbstractLabelMatrix {

        private:

            const uint8* y_;

        public:

            /**
             * @param numExamples   The number of examples
             * @param numLabels     The number of labels
             * @param y             A pointer to a C-contiguous array of type `uint8`, shape `(numExamples, numLabels)`,
             *                      representing the labels of the training examples
             */
            DenseLabelMatrixImpl(intp numExamples, intp numLabels, const uint8* y);

            ~DenseLabelMatrixImpl();

            uint8 getLabel(intp exampleIndex, intp labelIndex) override;

    };

    /**
     * Implements random access to the labels of the training examples based on a sparse matrix in the dictionary of
     * keys (DOK) format.
     */
    class DokLabelMatrixImpl : public AbstractLabelMatrix {

        private:

            sparse::BinaryDokMatrix* dokMatrix_;

        public:

            /**
             * @param numExamples   The number of examples
             * @param numLabels     The number of labels
             * @param dokMatrix     A pointer to an object of type `BinaryDokMatrix`, storing the relevant labels of the
             *                      training examples
             */
            DokLabelMatrixImpl(intp numExamples, intp numLabels, sparse::BinaryDokMatrix* dokMatrix);

            ~DokLabelMatrixImpl();

            uint8 getLabel(intp exampleIndex, intp labelIndex) override;

    };

    /**
     * An abstract base class for all classes that allow to search for the best refinement of a rule based on previously
     * stored statistics.
     */
    class AbstractRefinementSearch {

        public:

            virtual ~AbstractRefinementSearch();

            /**
             * Notifies the search that a specific statistic is covered by the condition that is currently considered
             * for refining a rule.
             *
             * This function must be called repeatedly for each statistic that is covered by the current condition,
             * immediately after the invocation of the function `Statistics#beginSearch`. Each of these statistics must
             * have been provided earlier via the function `Statistics#addSampledStatistic` or
             * `Statistics#updateCoveredStatistic`.
             *
             * This function is supposed to update any internal state of the search that relates to the examples that
             * are covered by the current condition, i.e., to compute and store local information that is required by
             * the other functions that will be called later. Any information computed by this function is expected to
             * be reset when invoking the function `resetSearch` for the next time.
             *
             * @param statistic_index   The index of the covered statistic
             * @param weight            The weight of the covered statistic
             */
            virtual void updateSearch(intp statisticIndex, uint32 weight);

            /**
             * Resets the internal state of the search that has been updated via preceding calls to the function
             * `updateSearch` to the state when the search was started via the function `Statistics#beginSearch`. When
             * calling this function, the current state must not be purged entirely, but it must be cached and made
             * available for use by the functions `calculateExampleWisePrediction` and `calculateLabelWisePrediction`
             * (if the function argument `accumulated` is set accordingly).
             *
             * This function may be invoked multiple times with one or several calls to `updateSearch` in between, which
             * is supposed to update the previously cached state by accumulating the current one, i.e., the accumulated
             * cached state should be the same as if `resetSearch` would not have been called at all.
             */
            virtual void resetSearch();

            /**
             * Calculates and returns the scores to be predicted by a rule that covers all statistics that have been
             * provided to the search so far via the function `updateSearch`.
             *
             * If the argument `uncovered` is 1, the rule is considered to cover all statistics that belong to the
             * difference between the statistics that have been provided via the function `addSampledStatistic` or
             * `updateCoveredStatistic` and the statistics that have been provided via the function `updateSearch`.
             *
             * If the argument `accumulated` is 1, all statistics that have been provided since the search has been
             * started via the function `Statistics#beginSearch` are taken into account even if the function
             * `resetSearch` has been called since then. If said function has not been invoked, this argument does not
             * have any effect.
             *
             * The calculated scores correspond to the subset of labels that have been provided when starting the search
             * via the function `Statistics#beginSearch`. The score to be predicted for an individual label is
             * calculated independently of the other labels, i.e., in the non-decomposable case it is assumed that the
             * rule will not predict for any other labels. In addition to each score, a quality score, which assesses
             * the quality of the prediction for the respective label, is returned.
             *
             * @param uncovered     0, if the rule covers all statistics that have been provided via the function
             *                      `updateSearch`, 1, if the rule covers all examples that belong to the difference
             *                      between the statistics that have been provided via the function
             *                      `Statistics#addSampledStatistic` or `Statistics#updateCoveredStatistic` and the
             *                      statistics that have been provided via the function `updateSearch`
             * @param accumulated   0, if the rule covers all statistics that have been provided via the function
             *                      `updateSearch` since the function `resetSearch` has been called for the last time,
             *                      1, if the rule covers all examples that have been provided since the search has been
             *                      started via the function `Statistics#beginSearch`
             * @return              A pointer to an object of type `LabelWisePrediction` that stores the scores to be
             *                      predicted by the rule for each considered label, as well as the corresponding
             *                      quality scores
             */
            virtual rule_evaluation::LabelWisePrediction* calculateLabelWisePrediction(bool uncovered,
                                                                                       bool accumulated);

    };

}
