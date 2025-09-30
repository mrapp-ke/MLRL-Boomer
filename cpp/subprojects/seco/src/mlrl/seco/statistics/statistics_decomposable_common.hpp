/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_sparse_array_binary.hpp"
#include "mlrl/seco/statistics/statistics_decomposable.hpp"
#include "statistics_common.hpp"
#include "statistics_state.hpp"

#include <memory>
#include <utility>

namespace seco {

    /**
     * An abstract base class for all statistics that provide access to the elements of confusion matrices that are
     * computed independently for each output.
     *
     * @tparam StatisticMatrix          The type of the matrix that provides access to the confusion matrices
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     */
    template<typename StatisticMatrix, typename RuleEvaluationFactory>
    class AbstractDecomposableStatistics
        : public AbstractCoverageStatistics<CoverageStatisticsState<StatisticMatrix>, RuleEvaluationFactory>,
          virtual public IDecomposableStatistics<RuleEvaluationFactory> {
        public:

            /**
             * @param statisticMatrixPtr    An unique pointer to an object of template type `StatisticMatrix` that
             *                              stores the confusion matrices
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that should be used for calculating
             *                              the predictions of rules, as well as corresponding quality scores
             */
            AbstractDecomposableStatistics(std::unique_ptr<StatisticMatrix> statisticMatrixPtr,
                                           const RuleEvaluationFactory& ruleEvaluationFactory)
                : AbstractCoverageStatistics<CoverageStatisticsState<StatisticMatrix>, RuleEvaluationFactory>(
                    std::make_unique<CoverageStatisticsState<StatisticMatrix>>(std::move(statisticMatrixPtr)),
                    ruleEvaluationFactory) {}

            /**
             * @see `IDecomposableStatistics::setRuleEvaluationFactory`
             */
            void setRuleEvaluationFactory(const RuleEvaluationFactory& ruleEvaluationFactory) override final {
                this->ruleEvaluationFactory_ = &ruleEvaluationFactory;
            }
    };

}
