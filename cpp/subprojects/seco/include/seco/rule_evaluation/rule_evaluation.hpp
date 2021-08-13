/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/rule_evaluation/score_vector.hpp"
#include "seco/data/vector_dense_confusion_matrices.hpp"


namespace seco {

    /**
     * Defines an interface for all classes that allow to calculate the predictions of rules, as well as corresponding
     * quality scores, based on confusion matrices.
     */
    class IRuleEvaluation {

        public:

            virtual ~IRuleEvaluation() { };

            /**
             * Calculates the scores to be predicted by a rule, as well as an overall quality score, based on label-wise
             * confusion matrices.
             *
             * @param majorityLabelVector       A reference to an object of type `DenseVector` that stores the
             *                                  predictions of the default rule
             * @param confusionMatricesTotal    A reference to an object of type `DenseConfusionMatrixVector` that
             *                                  stores confusion matrices that take into account all examples
             * @param confusionMatricesSubset   A reference to an object of type `DenseConfusionMatrixVector` that
             *                                  stores confusion matrices that take into account all all examples, which
             *                                  are covered by the previous refinement of the rule
             * @param confusionMatricesCovered  A reference to an object of type `DenseConfusionMatrixVector` that
             *                                  stores a confusion matrices that take into account all examples, which
             *                                  are covered by the rule
             * @param uncovered                 False, if the confusion matrices in `confusion_matrices_covered`
             *                                  correspond to the examples that are covered by rule, true, if they
             *                                  correspond to the examples that are not covered by the rule
             * @return                          A reference to an object of type `IScoreVector` that stores the
             *                                  predicted scores, as well as an overall quality score
             */
            virtual const IScoreVector& calculatePrediction(const BinarySparseArrayVector& majorityLabelVector,
                                                            const DenseConfusionMatrixVector& confusionMatricesTotal,
                                                            const DenseConfusionMatrixVector& confusionMatricesSubset,
                                                            const DenseConfusionMatrixVector& confusionMatricesCovered,
                                                            bool uncovered) = 0;

    };

}
