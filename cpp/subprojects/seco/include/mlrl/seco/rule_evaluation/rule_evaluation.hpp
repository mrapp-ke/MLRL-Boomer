/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/rule_evaluation/score_vector.hpp"
#include "mlrl/seco/data/vector_confusion_matrix_dense.hpp"

namespace seco {

    /**
     * Defines an interface for all classes that allow to calculate the predictions of rules, as well as their overall
     * quality, based on confusion matrices.
     */
    class IRuleEvaluation {
        public:

            virtual ~IRuleEvaluation() {}

            /**
             * Calculates the scores to be predicted by a rule, as well as their overall quality, based on confusion
             * matrices.
             *
             * @param majorityLabelIndicesBegin An iterator to the beginning of the indices of the labels that are
             *                                  relevant to the majority of the training examples
             * @param majorityLabelIndicesEnd   An iterator to the end of the indices of the labels that are relevant
             *                                  to the majority of the training examples
             * @param confusionMatricesTotal    A reference to an object of type `DenseConfusionMatrixVector` that
             *                                  stores confusion matrices that take into account all examples
             * @param confusionMatricesCovered  A reference to an object of type `DenseConfusionMatrixVector` that
             *                                  stores confusion matrices that take into account all examples, which are
             *                                  covered by the rule
             * @return                          A reference to an object of type `IScoreVector` that stores the
             *                                  predicted scores, as well as their overall quality
             */
            virtual const IScoreVector& calculateScores(View<uint32>::const_iterator majorityLabelIndicesBegin,
                                                        View<uint32>::const_iterator majorityLabelIndicesEnd,
                                                        const DenseConfusionMatrixVector& confusionMatricesTotal,
                                                        const DenseConfusionMatrixVector& confusionMatricesCovered) = 0;
    };

}
