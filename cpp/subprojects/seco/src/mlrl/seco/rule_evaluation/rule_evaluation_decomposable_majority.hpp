/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/iterator/iterator_forward_sparse_binary.hpp"
#include "mlrl/common/rule_evaluation/score_vector_dense.hpp"
#include "mlrl/seco/rule_evaluation/rule_evaluation_decomposable.hpp"

#include <memory>

#include <memory>

namespace seco {

    /**
     * Allows to calculate the predictions of rules, such that they predict each label as relevant or irrelevant,
     * depending on whether it is associated with the majority of the training examples or not.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class DecomposableMajorityRuleEvaluation final : public IRuleEvaluation {
        private:

            DenseScoreVector<T> scoreVector_;

        public:

            /**
             * @param labelIndices A reference to an object of template type `T` that provides access to the indices of
             *                     the labels for which the rules may predict
             */
            DecomposableMajorityRuleEvaluation(const T& labelIndices) : scoreVector_(labelIndices, true) {
                scoreVector_.quality = 0;
            }

            const IScoreVector& calculateScores(View<uint32>::const_iterator majorityLabelIndicesBegin,
                                                View<uint32>::const_iterator majorityLabelIndicesEnd,
                                                const DenseConfusionMatrixVector& confusionMatricesTotal,
                                                const DenseConfusionMatrixVector& confusionMatricesCovered) override {
                typename DenseScoreVector<T>::value_iterator valueIterator = scoreVector_.values_begin();
                typename DenseScoreVector<T>::index_const_iterator indexIterator = scoreVector_.indices_cbegin();
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

            std::unique_ptr<IRuleEvaluation> create(const CompleteIndexVector& indexVector) const override {
                return std::make_unique<DecomposableMajorityRuleEvaluation<CompleteIndexVector>>(indexVector);
            }

            std::unique_ptr<IRuleEvaluation> create(const PartialIndexVector& indexVector) const override {
                return std::make_unique<DecomposableMajorityRuleEvaluation<PartialIndexVector>>(indexVector);
            }
    };

}
