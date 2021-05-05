/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/measures/measure_evaluation.hpp"
#include "common/measures/measure_similarity.hpp"
#include "boosting/data/statistic_matrix_dense_example_wise.hpp"


namespace boosting {

    /**
     * Defines an interface for all (non-decomposable) loss functions that are applied example-wise.
     */
    class IExampleWiseLoss : public IEvaluationMeasure, public ISimilarityMeasure {

        public:

            virtual ~IExampleWiseLoss() { };

            /**
             * Updates the statistics of the example at a specific index.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `CContiguousLabelMatrix` that provides random
             *                          access to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `CContiguousConstView` that stores the 
             *                          currently predicted scores
             * @param statisticMatrix   A reference to an object of type `DenseExampleWiseStatisticMatrix` to be updated
             */
            virtual void updateExampleWiseStatistics(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                                     const CContiguousConstView<float64>& scoreMatrix,
                                                     DenseExampleWiseStatisticMatrix& statisticMatrix) const = 0;

             /**
             * Updates the statistics of the example at a specific index.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `CsrLabelMatrix` that provides row-wise access
             *                          to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `CContiguousConstView` that stores the
             *                          currently predicted scores
             * @param statisticMatrix   A reference to an object of type `DenseExampleWiseStatisticMatrix` to be updated
             */
            virtual void updateExampleWiseStatistics(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                                     const CContiguousConstView<float64>& scoreMatrix,
                                                     DenseExampleWiseStatisticMatrix& statisticMatrix) const = 0;

    };

}
