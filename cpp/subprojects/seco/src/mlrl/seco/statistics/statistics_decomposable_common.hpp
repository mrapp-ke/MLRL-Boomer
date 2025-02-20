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
     * @tparam LabelMatrix              The type of the matrix that provides access to the labels of the training
     *                                  examples
     * @tparam CoverageMatrix           The type of the matrix that is used to store how often individual examples and
     *                                  labels have been covered
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     */
    template<typename LabelMatrix, typename CoverageMatrix, typename RuleEvaluationFactory>
    class AbstractDecomposableStatistics
        : public AbstractStatistics<CoverageStatisticsState<LabelMatrix, CoverageMatrix>, RuleEvaluationFactory>,
          virtual public IDecomposableStatistics<RuleEvaluationFactory> {
        public:

            /**
             * @param labelMatrix               A reference to an object of template type `LabelMatrix` that provides
             *                                  access to the labels of the training examples
             * @param coverageMatrixPtr         An unique pointer to an object of template type `CoverageMatrix` that
             *                                  stores how often individual examples and labels have been covered
             * @param majorityLabelVectorPtr    An unique pointer to an object of type `BinarySparseArrayVector` that
             *                                  stores the predictions of the default rule
             * @param ruleEvaluationFactory     A reference to an object of template type `RuleEvaluationFactory` that
             *                                  allows to create instances of the class that should be used for
             *                                  calculating the predictions of rules, as well as corresponding quality
             *                                  scores
             */
            AbstractDecomposableStatistics(const LabelMatrix& labelMatrix,
                                           std::unique_ptr<CoverageMatrix> coverageMatrixPtr,
                                           std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr,
                                           const RuleEvaluationFactory& ruleEvaluationFactory)
                : AbstractStatistics<CoverageStatisticsState<LabelMatrix, CoverageMatrix>, RuleEvaluationFactory>(
                    std::make_unique<CoverageStatisticsState<LabelMatrix, CoverageMatrix>>(
                      labelMatrix, std::move(coverageMatrixPtr), std::move(majorityLabelVectorPtr)),
                    ruleEvaluationFactory) {}

            /**
             * @see `IDecomposableStatistics::setRuleEvaluationFactory`
             */
            void setRuleEvaluationFactory(const RuleEvaluationFactory& ruleEvaluationFactory) override final {
                this->ruleEvaluationFactory_ = &ruleEvaluationFactory;
            }
    };

}
