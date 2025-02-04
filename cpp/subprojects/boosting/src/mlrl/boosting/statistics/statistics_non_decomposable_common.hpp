/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/statistics/statistics_non_decomposable.hpp"
#include "statistics_common.hpp"

#include <memory>
#include <utility>

namespace boosting {

    template<typename OutputMatrix, typename StatisticView, typename ScoreMatrix, typename LossFunction>
    static inline void updateNonDecomposableStatisticsInternally(uint32 statisticIndex,
                                                                 const OutputMatrix& outputMatrix,
                                                                 StatisticView& statisticView, ScoreMatrix& scoreMatrix,
                                                                 const LossFunction& lossFunction) {
        lossFunction.updateNonDecomposableStatistics(statisticIndex, outputMatrix, scoreMatrix.getView(),
                                                     statisticView);
    }

    /**
     * An abstract base class for all statistics that provide access to gradients and Hessians that are calculated
     * according to a non-decomposable loss function.
     *
     * @tparam OutputMatrix                         The type of the matrix that provides access to the ground truth of
     *                                              the training examples
     * @tparam StatisticMatrix                      The type of the matrix that stores the gradients and Hessians
     * @tparam ScoreMatrix                          The type of the matrices that are used to store predicted scores
     * @tparam LossFunction                         The type of the loss function that is used to calculate gradients
     *                                              and Hessians
     * @tparam EvaluationMeasure                    The type of the evaluation measure that is used to assess the
     *                                              quality of predictions for a specific statistic
     * @tparam NonDecomposableRuleEvaluationFactory The type of the factory that allows to create instances of the class
     *                                              that is used for calculating the predictions of rules, as well as
     *                                              their overall quality, based on gradients and Hessians that have
     *                                              been calculated according to a non-decomposable loss function
     * @tparam DecomposableRuleEvaluationFactory    The type of the factory that allows to create instances of the class
     *                                              that is used for calculating the predictions of rules, as well as
     *                                              their overall quality, based on gradients and Hessians that have
     *                                              been calculated according to a decomposable loss function
     */
    template<typename OutputMatrix, typename StatisticMatrix, typename ScoreMatrix, typename LossFunction,
             typename EvaluationMeasure, typename NonDecomposableRuleEvaluationFactory,
             typename DecomposableRuleEvaluationFactory>
    class AbstractNonDecomposableStatistics
        : public AbstractStatistics<OutputMatrix, StatisticMatrix, ScoreMatrix, LossFunction, EvaluationMeasure,
                                    NonDecomposableRuleEvaluationFactory>,
          virtual public INonDecomposableStatistics<NonDecomposableRuleEvaluationFactory,
                                                    DecomposableRuleEvaluationFactory> {
        protected:

            void updateStatistics(uint32 statisticIndex, const CompletePrediction& prediction) override final {
                updateNonDecomposableStatisticsInternally(statisticIndex, this->outputMatrix_,
                                                          this->statisticMatrixPtr_->getView(),
                                                          *(this->scoreMatrixPtr_), *(this->lossPtr_));
            }

            void updateStatistics(uint32 statisticIndex, const PartialPrediction& prediction) override final {
                updateNonDecomposableStatisticsInternally(statisticIndex, this->outputMatrix_,
                                                          this->statisticMatrixPtr_->getView(),
                                                          *(this->scoreMatrixPtr_), *(this->lossPtr_));
            }

        public:

            /**
             * @param lossPtr               An unique pointer to an object of template type `LossFunction` that
             *                              implements the loss function that should be used for calculating gradients
             *                              and Hessians
             * @param evaluationMeasurePtr  An unique pointer to an object of template type `EvaluationMeasure` that
             *                              implements the evaluation measure that should be used to assess the quality
             *                              of predictions for a specific statistic
             * @param ruleEvaluationFactory A reference to an object of template type
             *                              `NonDecomposableRuleEvaluationFactory` that allows to create instances of
             *                              the class that should be used for calculating the predictions of rules, as
             *                              well as their overall quality
             * @param outputMatrix          A reference to an object of template type `OutputMatrix` that provides
             *                              access to the ground truth of the training examples
             * @param statisticMatrixPtr    An unique pointer to an object of template type `StatisticView` that stores
             *                              the gradients and Hessians
             * @param scoreMatrixPtr        An unique pointer to an object of template type `ScoreMatrix` that stores
             *                              the currently predicted scores
             */
            AbstractNonDecomposableStatistics(std::unique_ptr<LossFunction> lossPtr,
                                              std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr,
                                              const NonDecomposableRuleEvaluationFactory& ruleEvaluationFactory,
                                              const OutputMatrix& outputMatrix,
                                              std::unique_ptr<StatisticMatrix> statisticMatrixPtr,
                                              std::unique_ptr<ScoreMatrix> scoreMatrixPtr)
                : AbstractStatistics<OutputMatrix, StatisticMatrix, ScoreMatrix, LossFunction, EvaluationMeasure,
                                     NonDecomposableRuleEvaluationFactory>(
                    std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, outputMatrix,
                    std::move(statisticMatrixPtr), std::move(scoreMatrixPtr)) {}

            /**
             * @see `INonDecomposableStatistics::setRuleEvaluationFactory`
             */
            void setRuleEvaluationFactory(
              const NonDecomposableRuleEvaluationFactory& ruleEvaluationFactory) override final {
                this->ruleEvaluationFactory_ = &ruleEvaluationFactory;
            }
    };

}
