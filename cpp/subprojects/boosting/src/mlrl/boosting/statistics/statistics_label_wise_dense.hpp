/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/matrix_c_contiguous_numeric.hpp"
#include "mlrl/boosting/data/statistic_vector_label_wise_dense.hpp"
#include "mlrl/boosting/losses/loss_label_wise.hpp"
#include "mlrl/common/measures/measure_evaluation.hpp"
#include "statistics_label_wise_common.hpp"

namespace boosting {

    /**
     * A matrix that stores gradients and Hessians that have been calculated using a label-wise decomposable loss
     * function using C-contiguous arrays.
     */
    class DenseLabelWiseStatisticMatrix final : public CContiguousView<Tuple<float64>> {
        public:

            /**
             * @param numRows   The number of rows in the matrix
             * @param numCols   The number of columns in the matrix
             */
            DenseLabelWiseStatisticMatrix(uint32 numRows, uint32 numCols)
                : CContiguousView<Tuple<float64>>(allocateMemory<Tuple<float64>>(numRows * numCols), numRows, numCols) {
            }

            ~DenseLabelWiseStatisticMatrix() override {
                freeMemory(this->array);
            }

            /**
             * Adds all gradients and Hessians in a vector to a specific row of this matrix. The gradients and Hessians
             * to be added are multiplied by a specific weight.
             *
             * @param row       The row
             * @param begin     An iterator to the beginning of the vector
             * @param end       An iterator to the end of the vector
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToRow(uint32 row, value_const_iterator begin, value_const_iterator end, float64 weight) {
                addToView(CContiguousView::values_begin(row), begin, Matrix::numCols, weight);
            }

            /**
             * Returns the number of rows in the view.
             *
             * @return The number of rows
             */
            uint32 getNumRows() const {
                return Matrix::numRows;
            }

            /**
             * Returns the number of columns in the view.
             *
             * @return The number of columns
             */
            uint32 getNumCols() const {
                return Matrix::numCols;
            }
    };

    /**
     * Provides access to gradients and Hessians that have been calculated according to a differentiable loss function
     * that is applied label-wise and are stored using dense data structures.
     *
     * @tparam LabelMatrix The type of the matrix that provides access to the labels of the training examples
     */
    template<typename LabelMatrix>
    class DenseLabelWiseStatistics final
        : public AbstractLabelWiseStatistics<LabelMatrix, DenseLabelWiseStatisticVector, DenseLabelWiseStatisticMatrix,
                                             DenseLabelWiseStatisticMatrix, NumericCContiguousMatrix<float64>,
                                             ILabelWiseLoss, IEvaluationMeasure, ILabelWiseRuleEvaluationFactory> {
        public:

            /**
             * @param lossPtr               An unique pointer to an object of type `ILabelWiseLoss` that implements the
             *                              loss function that should be used for calculating gradients and Hessians
             * @param evaluationMeasurePtr  An unique pointer to an object of type `IEvaluationMeasure` that implements
             *                              the evaluation measure that should be used to assess the quality of
             *                              predictions for a specific statistic
             * @param ruleEvaluationFactory A reference to an object of type `ILabelWiseRuleEvaluationFactory`, that
             *                              allows to create instances of the class that is used for calculating the
             *                              predictions of rules, as well as their overall quality
             * @param labelMatrix           A reference to an object of template type `LabelMatrix` that provides access
             *                              to the labels of the training examples
             * @param statisticViewPtr      An unique pointer to an object of type `DenseLabelWiseStatisticMatrix` that
             *                              provides access to the gradients and Hessians
             * @param scoreMatrixPtr        An unique pointer to an object of type `NumericCContiguousMatrix` that
             *                              stores the currently predicted scores
             */
            DenseLabelWiseStatistics(std::unique_ptr<ILabelWiseLoss> lossPtr,
                                     std::unique_ptr<IEvaluationMeasure> evaluationMeasurePtr,
                                     const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory,
                                     const LabelMatrix& labelMatrix,
                                     std::unique_ptr<DenseLabelWiseStatisticMatrix> statisticViewPtr,
                                     std::unique_ptr<NumericCContiguousMatrix<float64>> scoreMatrixPtr)
                : AbstractLabelWiseStatistics<LabelMatrix, DenseLabelWiseStatisticVector, DenseLabelWiseStatisticMatrix,
                                              DenseLabelWiseStatisticMatrix, NumericCContiguousMatrix<float64>,
                                              ILabelWiseLoss, IEvaluationMeasure, ILabelWiseRuleEvaluationFactory>(
                  std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, labelMatrix,
                  std::move(statisticViewPtr), std::move(scoreMatrixPtr)) {}

            /**
             * @see `IBoostingStatistics::visitScoreMatrix`
             */
            void visitScoreMatrix(IBoostingStatistics::DenseScoreMatrixVisitor denseVisitor,
                                  IBoostingStatistics::SparseScoreMatrixVisitor sparseVisitor) const override {
                denseVisitor(this->scoreMatrixPtr_->getView());
            }
    };

}
