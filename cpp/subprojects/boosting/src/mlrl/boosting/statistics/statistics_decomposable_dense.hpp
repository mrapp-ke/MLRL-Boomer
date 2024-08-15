/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/matrix_c_contiguous_numeric.hpp"
#include "mlrl/boosting/data/vector_statistic_decomposable_dense.hpp"
#include "mlrl/boosting/losses/loss_decomposable.hpp"
#include "mlrl/common/measures/measure_evaluation.hpp"
#include "statistics_decomposable_common.hpp"

#include <memory>
#include <utility>

namespace boosting {

    /**
     * A matrix that stores gradients and Hessians that have been calculated using a decomposable loss function using
     * C-contiguous arrays.
     */
    class DenseDecomposableStatisticMatrix final
        : public ClearableViewDecorator<MatrixDecorator<AllocatedCContiguousView<Tuple<float64>>>> {
        public:

            /**
             * @param numRows   The number of rows in the matrix
             * @param numCols   The number of columns in the matrix
             */
            DenseDecomposableStatisticMatrix(uint32 numRows, uint32 numCols)
                : ClearableViewDecorator<MatrixDecorator<AllocatedCContiguousView<Tuple<float64>>>>(
                    AllocatedCContiguousView<Tuple<float64>>(numRows, numCols)) {}

            /**
             * Adds all gradients and Hessians in a vector to a specific row of this matrix. The gradients and Hessians
             * to be added are multiplied by a specific weight.
             *
             * @param row       The row
             * @param begin     An iterator to the beginning of the vector
             * @param end       An iterator to the end of the vector
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToRow(uint32 row, View<Tuple<float64>>::const_iterator begin,
                          View<Tuple<float64>>::const_iterator end, float64 weight) {
                util::addToView(this->view.values_begin(row), begin, this->getNumCols(), weight);
            }
    };

    /**
     * Provides access to gradients and Hessians that have been calculated according to a decomposable loss function and
     * are stored using dense data structures.
     *
     * @tparam Loss                 The type of the loss function
     * @tparam OutputMatrix         The type of the matrix that provides access to the ground truth of the training
     *                              examples
     * @tparam EvaluationMeasure    The type of the evaluation that should be used to access the quality of predictions
     */
    template<typename Loss, typename OutputMatrix, typename EvaluationMeasure>
    class DenseDecomposableStatistics final
        : public AbstractDecomposableStatistics<OutputMatrix, DenseDecomposableStatisticVector,
                                                DenseDecomposableStatisticMatrix, NumericCContiguousMatrix<float64>,
                                                Loss, EvaluationMeasure, IDecomposableRuleEvaluationFactory> {
        public:

            /**
             * @param lossPtr               An unique pointer to an object of template type `Loss` that implements the
             *                              loss function that should be used for calculating gradients and Hessians
             * @param evaluationMeasurePtr  An unique pointer to an object of template type `EvaluationMeasure` that
             *                              implements the evaluation measure that should be used to assess the quality
             *                              of predictions for a specific statistic
             * @param ruleEvaluationFactory A reference to an object of type `IDecomposableRuleEvaluationFactory`, that
             *                              allows to create instances of the class that is used for calculating the
             *                              predictions of rules, as well as their overall quality
             * @param outputMatrix          A reference to an object of template type `OutputMatrix` that provides
             *                              access to the ground truth of the training examples
             * @param statisticMatrixPtr    An unique pointer to an object of type `DenseDecomposableStatisticMatrix`
             *                              that provides access to the gradients and Hessians
             * @param scoreMatrixPtr        An unique pointer to an object of type `NumericCContiguousMatrix` that
             *                              stores the currently predicted scores
             */
            DenseDecomposableStatistics(std::unique_ptr<Loss> lossPtr,
                                        std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr,
                                        const IDecomposableRuleEvaluationFactory& ruleEvaluationFactory,
                                        const OutputMatrix& outputMatrix,
                                        std::unique_ptr<DenseDecomposableStatisticMatrix> statisticMatrixPtr,
                                        std::unique_ptr<NumericCContiguousMatrix<float64>> scoreMatrixPtr)
                : AbstractDecomposableStatistics<OutputMatrix, DenseDecomposableStatisticVector,
                                                 DenseDecomposableStatisticMatrix, NumericCContiguousMatrix<float64>,
                                                 Loss, EvaluationMeasure, IDecomposableRuleEvaluationFactory>(
                    std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, outputMatrix,
                    std::move(statisticMatrixPtr), std::move(scoreMatrixPtr)) {}

            /**
             * @see `IBoostingStatistics::visitScoreMatrix`
             */
            void visitScoreMatrix(IBoostingStatistics::DenseScoreMatrixVisitor denseVisitor,
                                  IBoostingStatistics::SparseScoreMatrixVisitor sparseVisitor) const override {
                denseVisitor(this->scoreMatrixPtr_->getView());
            }
    };

}
