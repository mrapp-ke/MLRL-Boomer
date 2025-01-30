/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/iterator/iterator_forward_sparse_binary.hpp"
#include "mlrl/common/rule_evaluation/score_vector_dense.hpp"
#include "mlrl/seco/rule_evaluation/rule_evaluation_decomposable.hpp"

#include <memory>

namespace seco {

    /**
     * Allows to calculate the predictions of rules, based on confusion matrices, such that they predict each label as
     * relevant or irrelevant, depending on whether it is associated with the majority of the training examples or not.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the confusion matrices
     * @tparam IndexVector      The type of the vector that provides access to the labels for which predictions should
     *                          be calculated
     */
    template<typename StatisticVector, typename IndexVector>
    class DecomposableMajorityRuleEvaluation final : public IRuleEvaluation<StatisticVector> {
        private:

            DenseScoreVector<IndexVector> scoreVector_;

        public:

            /**
             * @param labelIndices A reference to an object of template type `IndexVector` that provides access to the
             *                     indices of the labels for which the rules may predict
             */
            DecomposableMajorityRuleEvaluation(const IndexVector& labelIndices) : scoreVector_(labelIndices, true) {
                scoreVector_.quality = 0;
            }

            const IScoreVector& calculateScores(View<uint32>::const_iterator majorityLabelIndicesBegin,
                                                View<uint32>::const_iterator majorityLabelIndicesEnd,
                                                const StatisticVector& confusionMatricesTotal,
                                                const StatisticVector& confusionMatricesCovered) override {
                typename DenseScoreVector<IndexVector>::value_iterator valueIterator = scoreVector_.values_begin();
                typename DenseScoreVector<IndexVector>::index_const_iterator indexIterator =
                  scoreVector_.indices_cbegin();
                auto labelIterator =
                  createBinarySparseForwardIterator(majorityLabelIndicesBegin, majorityLabelIndicesEnd);
                uint32 numElements = scoreVector_.getNumElements();
                uint32 previousIndex = 0;

                for (uint32 i = 0; i < numElements; i++) {
                    uint32 index = indexIterator[i];
                    std::advance(labelIterator, index - previousIndex);
                    valueIterator[i] = (float64) *labelIterator;
                    previousIndex = index;
                }

                return scoreVector_;
            }
    };

    /**
     * Allows to create instances of the class `DecomposableMajorityRuleEvaluation`.
     */
    class DecomposableMajorityRuleEvaluationFactory final : public IDecomposableRuleEvaluationFactory {
        public:

            std::unique_ptr<IRuleEvaluation<DenseConfusionMatrixVector<uint32>>> create(
              const DenseConfusionMatrixVector<uint32>& statisticVector,
              const CompleteIndexVector& indexVector) const override {
                return std::make_unique<
                  DecomposableMajorityRuleEvaluation<DenseConfusionMatrixVector<uint32>, CompleteIndexVector>>(
                  indexVector);
            }

            std::unique_ptr<IRuleEvaluation<DenseConfusionMatrixVector<uint32>>> create(
              const DenseConfusionMatrixVector<uint32>& statisticVector,
              const PartialIndexVector& indexVector) const override {
                return std::make_unique<
                  DecomposableMajorityRuleEvaluation<DenseConfusionMatrixVector<uint32>, PartialIndexVector>>(
                  indexVector);
            }

            std::unique_ptr<IRuleEvaluation<DenseConfusionMatrixVector<float32>>> create(
              const DenseConfusionMatrixVector<float32>& statisticVector,
              const CompleteIndexVector& indexVector) const override {
                return std::make_unique<
                  DecomposableMajorityRuleEvaluation<DenseConfusionMatrixVector<float32>, CompleteIndexVector>>(
                  indexVector);
            }

            std::unique_ptr<IRuleEvaluation<DenseConfusionMatrixVector<float32>>> create(
              const DenseConfusionMatrixVector<float32>& statisticVector,
              const PartialIndexVector& indexVector) const override {
                return std::make_unique<
                  DecomposableMajorityRuleEvaluation<DenseConfusionMatrixVector<float32>, PartialIndexVector>>(
                  indexVector);
            }
    };

}
