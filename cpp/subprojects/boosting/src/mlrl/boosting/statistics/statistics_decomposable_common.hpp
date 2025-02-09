/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/statistics/statistics_decomposable.hpp"
#include "statistics_common.hpp"
#include "statistics_state_decomposable.hpp"

namespace boosting {

    /**
     * An abstract base class for all statistics that provide access to gradients and Hessians that are calculated
     * according to a decomposable loss function.
     *
     * @tparam OutputMatrix             The type of the matrix that provides access to the ground truth of the training
     *                                  examples
     * @tparam StatisticMatrix          The type of the matrix that provides access to the gradients and Hessians
     * @tparam ScoreMatrix              The type of the matrices that are used to store predicted scores
     * @tparam LossFunction             The type of the loss function that is used to calculate gradients and Hessians
     * @tparam EvaluationMeasure        The type of the evaluation measure that is used to assess the quality of
     *                                  predictions for a specific statistic
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     */
    template<typename OutputMatrix, typename StatisticMatrix, typename ScoreMatrix, typename LossFunction,
             typename EvaluationMeasure, typename RuleEvaluationFactory>
    class AbstractDecomposableStatistics
        : public AbstractStatistics<
            DecomposableStatisticsState<OutputMatrix, StatisticMatrix, ScoreMatrix, LossFunction>, EvaluationMeasure,
            RuleEvaluationFactory>,
          virtual public IDecomposableStatistics<RuleEvaluationFactory> {
        public:

            /**
             * @param lossPtr               An unique pointer to an object of template type `LossFunction` that
             *                              implements the loss function that should be used for calculating gradients
             *                              and Hessians
             * @param evaluationMeasurePtr  An unique pointer to an object of template type `EvaluationMeasure` that
             *                              implements the evaluation measure that should be used to assess the quality
             *                              of predictions for a specific statistic
             * @param ruleEvaluationFactory A reference to an object of type `RuleEvaluationFactory` that allows to
             *                              create instances of the class that should be used for calculating the
             *                              predictions of rules, as well as their overall quality
             * @param outputMatrix          A reference to an object of template type `OutputMatrix` that provides
             *                              access to the ground truth of the training examples
             * @param statisticMatrixPtr    An unique pointer to an object of template type `StatisticMatrix` that
             *                              provides access to the gradients and Hessians
             * @param scoreMatrixPtr        An unique pointer to an object of template type `ScoreMatrix` that stores
             *                              the currently predicted scores
             */
            AbstractDecomposableStatistics(std::unique_ptr<LossFunction> lossPtr,
                                           std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr,
                                           const RuleEvaluationFactory& ruleEvaluationFactory,
                                           const OutputMatrix& outputMatrix,
                                           std::unique_ptr<StatisticMatrix> statisticMatrixPtr,
                                           std::unique_ptr<ScoreMatrix> scoreMatrixPtr)
                : AbstractStatistics<
                    DecomposableStatisticsState<OutputMatrix, StatisticMatrix, ScoreMatrix, LossFunction>,
                    EvaluationMeasure, RuleEvaluationFactory>(
                    std::make_unique<
                      DecomposableStatisticsState<OutputMatrix, StatisticMatrix, ScoreMatrix, LossFunction>>(
                      outputMatrix, std::move(statisticMatrixPtr), std::move(scoreMatrixPtr), std::move(lossPtr)),
                    std::move(evaluationMeasurePtr), ruleEvaluationFactory) {}

            /**
             * @see `IDecomposableStatistics::setRuleEvaluationFactory`
             */
            void setRuleEvaluationFactory(const RuleEvaluationFactory& ruleEvaluationFactory) override final {
                this->ruleEvaluationFactory_ = &ruleEvaluationFactory;
            }
    };

}
