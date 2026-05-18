#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_partial_dynamic.hpp"

#include "mlrl/boosting/rule_evaluation/simd/vector_math_decomposable_simd.hpp"
#include "mlrl/boosting/rule_evaluation/vector_math_decomposable.hpp"
#include "rule_evaluation_decomposable_complete_common.hpp"
#include "rule_evaluation_decomposable_partial_dynamic_common.hpp"

namespace boosting {

    /**
     * Allows to calculate the predictions of partial rules, which predict for a subset of the available outputs that is
     * determined dynamically, as well as their overall quality, based on the gradients and Hessians that are stored by
     * a vector using L1 and L2 regularization.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam IndexVector      The type of the vector that provides access to the indices of the outputs for which
     *                          predictions should be calculated
     * @tparam VectorMath       The type that implements basic operations for calculating with gradients and Hessians
     */
    template<typename StatisticVector, typename IndexVector, typename VectorMath>
    class DecomposableDynamicPartialRuleEvaluation final : public IRuleEvaluation<StatisticVector> {
        private:

            using statistic_type = StatisticVector::statistic_type;

            const IndexVector& outputIndices_;

            PartialIndexVector indexVector_;

            DenseScoreVector<statistic_type, PartialIndexVector> scoreVector_;

            const float32 threshold_;

            const float32 exponent_;

            const float32 l1RegularizationWeight_;

            const float32 l2RegularizationWeight_;

            template<typename StatisticType, typename WeightType>
            static inline void calculateScoresInternally(
              const SparseDecomposableStatisticVectorView<StatisticType, WeightType>& statisticVector,
              const IndexVector& outputIndices, DenseScoreVector<StatisticType, PartialIndexVector>& scoreVector,
              PartialIndexVector& indexVector, float32 l1RegularizationWeight, float32 l2RegularizationWeight,
              float32 threshold, float32 exponent) {
                uint32 numElements = statisticVector.getNumElements();
                auto gradientIterator = statisticVector.gradients_cbegin();
                auto hessianIterator = statisticVector.hessians_cbegin();
                const std::pair<statistic_type, statistic_type> pair = getMinAndMaxScore(
                  gradientIterator, hessianIterator, numElements, l1RegularizationWeight, l2RegularizationWeight);
                statistic_type minAbsScore = pair.first;
                statistic_type scoreThreshold = calculateThreshold(minAbsScore, pair.second, threshold, exponent);
                auto indexIterator = indexVector.begin();
                auto valueIterator = scoreVector.values_begin();
                auto outputIndexIterator = outputIndices.cbegin();
                statistic_type quality = 0;
                uint32 n = 0;

                for (uint32 i = 0; i < numElements; i++) {
                    statistic_type gradient = gradientIterator[i];
                    statistic_type hessian = hessianIterator[i];
                    statistic_type score =
                      calculateOutputWiseScore(gradient, hessian, l1RegularizationWeight, l2RegularizationWeight);

                    if (calculateWeightedScore(score, minAbsScore, exponent) >= scoreThreshold) {
                        indexIterator[n] = outputIndexIterator[i];
                        valueIterator[n] = score;
                        quality += calculateOutputWiseQuality(score, gradient, hessian, l1RegularizationWeight,
                                                              l2RegularizationWeight);
                        n++;
                    }
                }

                indexVector.setNumElements(n, false);
                scoreVector.quality = quality;
            }

            template<typename StatisticType>
            static inline void calculateScoresInternally(
              const DenseDecomposableStatisticVectorView<StatisticType>& statisticVector,
              const IndexVector& outputIndices, DenseScoreVector<StatisticType, PartialIndexVector>& scoreVector,
              PartialIndexVector& indexVector, float32 l1RegularizationWeight, float32 l2RegularizationWeight,
              float32 threshold, float32 exponent) {
                uint32 numElements = statisticVector.getNumElements();
                auto gradientIterator = statisticVector.gradients_cbegin();
                auto hessianIterator = statisticVector.hessians_cbegin();
                auto valueIterator = scoreVector.values_begin();

                VectorMath::calculateOutputWiseScores(gradientIterator, hessianIterator, valueIterator, numElements,
                                                      l1RegularizationWeight, l2RegularizationWeight);

                const std::pair<statistic_type, statistic_type> pair = getMinAndMaxScore(
                  gradientIterator, hessianIterator, numElements, l1RegularizationWeight, l2RegularizationWeight);
                statistic_type minAbsScore = pair.first;
                statistic_type scoreThreshold = calculateThreshold(minAbsScore, pair.second, threshold, exponent);
                auto indexIterator = indexVector.begin();
                auto outputIndexIterator = outputIndices.cbegin();
                statistic_type quality = 0;
                uint32 n = 0;

                for (uint32 i = 0; i < numElements; i++) {
                    statistic_type score = valueIterator[i];

                    if (calculateWeightedScore(score, minAbsScore, exponent) >= scoreThreshold) {
                        indexIterator[n] = outputIndexIterator[i];
                        valueIterator[n] = score;
                        quality += calculateOutputWiseQuality(score, gradientIterator[i], hessianIterator[i],
                                                              l1RegularizationWeight, l2RegularizationWeight);
                        n++;
                    }
                }

                indexVector.setNumElements(n, false);
                scoreVector.quality = quality;
            }

        public:

            /**
             * @param outputIndices             A reference to an object of template type `IndexVector` that provides
             *                                  access to the indices of the outputs for which the rules may predict
             * @param threshold                 A threshold that affects for how many outputs the rule heads should
             *                                  predict
             * @param exponent                  An exponent that is used to weigh that estimated predictive quality for
             *                                  individual outputs
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            DecomposableDynamicPartialRuleEvaluation(const IndexVector& outputIndices, float32 threshold,
                                                     float32 exponent, float32 l1RegularizationWeight,
                                                     float32 l2RegularizationWeight)
                : outputIndices_(outputIndices), indexVector_(outputIndices.getNumElements()),
                  scoreVector_(indexVector_, true), threshold_(1.0f - threshold), exponent_(exponent),
                  l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {}

            const IScoreVector& calculateScores(StatisticVector& statisticVector) override {
                calculateScoresInternally(statisticVector, outputIndices_, scoreVector_, indexVector_,
                                          l1RegularizationWeight_, l2RegularizationWeight_, threshold_, exponent_);
                return scoreVector_;
            }
    };

    template<typename VectorMath>
    DecomposableDynamicPartialRuleEvaluationFactory<VectorMath>::DecomposableDynamicPartialRuleEvaluationFactory(
      float32 threshold, float32 exponent, float32 l1RegularizationWeight, float32 l2RegularizationWeight)
        : threshold_(threshold), exponent_(exponent), l1RegularizationWeight_(l1RegularizationWeight),
          l2RegularizationWeight_(l2RegularizationWeight) {}

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float32>>>
      DecomposableDynamicPartialRuleEvaluationFactory<VectorMath>::create(
        const DenseDecomposableStatisticVectorView<float32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<DecomposableDynamicPartialRuleEvaluation<DenseDecomposableStatisticVectorView<float32>,
                                                                         CompleteIndexVector, VectorMath>>(
          indexVector, threshold_, exponent_, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float32>>>
      DecomposableDynamicPartialRuleEvaluationFactory<VectorMath>::create(
        const DenseDecomposableStatisticVectorView<float32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        if (indexVector.getNumElements() > 1) {
            return std::make_unique<DecomposableCompleteRuleEvaluation<DenseDecomposableStatisticVectorView<float32>,
                                                                       PartialIndexVector, VectorMath>>(
              indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
        }

        return std::make_unique<DecomposableCompleteRuleEvaluation<
          DenseDecomposableStatisticVectorView<float32>, PartialIndexVector, SequentialDecomposableVectorMath>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float64>>>
      DecomposableDynamicPartialRuleEvaluationFactory<VectorMath>::create(
        const DenseDecomposableStatisticVectorView<float64>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<DecomposableDynamicPartialRuleEvaluation<DenseDecomposableStatisticVectorView<float64>,
                                                                         CompleteIndexVector, VectorMath>>(
          indexVector, threshold_, exponent_, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float64>>>
      DecomposableDynamicPartialRuleEvaluationFactory<VectorMath>::create(
        const DenseDecomposableStatisticVectorView<float64>& statisticVector,
        const PartialIndexVector& indexVector) const {
        if (indexVector.getNumElements() > 1) {
            return std::make_unique<DecomposableCompleteRuleEvaluation<DenseDecomposableStatisticVectorView<float64>,
                                                                       PartialIndexVector, VectorMath>>(
              indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
        }

        return std::make_unique<DecomposableCompleteRuleEvaluation<
          DenseDecomposableStatisticVectorView<float64>, PartialIndexVector, SequentialDecomposableVectorMath>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVectorView<float32, uint32>>>
      DecomposableDynamicPartialRuleEvaluationFactory<VectorMath>::create(
        const SparseDecomposableStatisticVectorView<float32, uint32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<DecomposableDynamicPartialRuleEvaluation<
          SparseDecomposableStatisticVectorView<float32, uint32>, CompleteIndexVector, VectorMath>>(
          indexVector, threshold_, exponent_, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVectorView<float32, uint32>>>
      DecomposableDynamicPartialRuleEvaluationFactory<VectorMath>::create(
        const SparseDecomposableStatisticVectorView<float32, uint32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        if (indexVector.getNumElements() > 1) {
            return std::make_unique<DecomposableCompleteRuleEvaluation<
              SparseDecomposableStatisticVectorView<float32, uint32>, PartialIndexVector, VectorMath>>(
              indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
        }

        return std::make_unique<
          DecomposableCompleteRuleEvaluation<SparseDecomposableStatisticVectorView<float32, uint32>, PartialIndexVector,
                                             SequentialDecomposableVectorMath>>(indexVector, l1RegularizationWeight_,
                                                                                l2RegularizationWeight_);
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVectorView<float32, float32>>>
      DecomposableDynamicPartialRuleEvaluationFactory<VectorMath>::create(
        const SparseDecomposableStatisticVectorView<float32, float32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<DecomposableDynamicPartialRuleEvaluation<
          SparseDecomposableStatisticVectorView<float32, float32>, CompleteIndexVector, VectorMath>>(
          indexVector, threshold_, exponent_, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVectorView<float32, float32>>>
      DecomposableDynamicPartialRuleEvaluationFactory<VectorMath>::create(
        const SparseDecomposableStatisticVectorView<float32, float32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        if (indexVector.getNumElements() > 1) {
            return std::make_unique<DecomposableCompleteRuleEvaluation<
              SparseDecomposableStatisticVectorView<float32, float32>, PartialIndexVector, VectorMath>>(
              indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
        }

        return std::make_unique<
          DecomposableCompleteRuleEvaluation<SparseDecomposableStatisticVectorView<float32, float32>,
                                             PartialIndexVector, SequentialDecomposableVectorMath>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVectorView<float64, uint32>>>
      DecomposableDynamicPartialRuleEvaluationFactory<VectorMath>::create(
        const SparseDecomposableStatisticVectorView<float64, uint32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<DecomposableDynamicPartialRuleEvaluation<
          SparseDecomposableStatisticVectorView<float64, uint32>, CompleteIndexVector, VectorMath>>(
          indexVector, threshold_, exponent_, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVectorView<float64, uint32>>>
      DecomposableDynamicPartialRuleEvaluationFactory<VectorMath>::create(
        const SparseDecomposableStatisticVectorView<float64, uint32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        if (indexVector.getNumElements() > 1) {
            return std::make_unique<DecomposableCompleteRuleEvaluation<
              SparseDecomposableStatisticVectorView<float64, uint32>, PartialIndexVector, VectorMath>>(
              indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
        }

        return std::make_unique<
          DecomposableCompleteRuleEvaluation<SparseDecomposableStatisticVectorView<float64, uint32>, PartialIndexVector,
                                             SequentialDecomposableVectorMath>>(indexVector, l1RegularizationWeight_,
                                                                                l2RegularizationWeight_);
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVectorView<float64, float32>>>
      DecomposableDynamicPartialRuleEvaluationFactory<VectorMath>::create(
        const SparseDecomposableStatisticVectorView<float64, float32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<DecomposableDynamicPartialRuleEvaluation<
          SparseDecomposableStatisticVectorView<float64, float32>, CompleteIndexVector, VectorMath>>(
          indexVector, threshold_, exponent_, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVectorView<float64, float32>>>
      DecomposableDynamicPartialRuleEvaluationFactory<VectorMath>::create(
        const SparseDecomposableStatisticVectorView<float64, float32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        if (indexVector.getNumElements() > 1) {
            return std::make_unique<DecomposableCompleteRuleEvaluation<
              SparseDecomposableStatisticVectorView<float64, float32>, PartialIndexVector, VectorMath>>(
              indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
        }

        return std::make_unique<
          DecomposableCompleteRuleEvaluation<SparseDecomposableStatisticVectorView<float64, float32>,
                                             PartialIndexVector, SequentialDecomposableVectorMath>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    template class DecomposableDynamicPartialRuleEvaluationFactory<SequentialDecomposableVectorMath>;
#if SIMD_SUPPORT_ENABLED
    template class DecomposableDynamicPartialRuleEvaluationFactory<SimdDecomposableVectorMath>;
#endif
}
