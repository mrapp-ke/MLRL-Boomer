#include "boosting/rule_evaluation/rule_evaluation_example_wise_partial_fixed_binned.hpp"
#include "rule_evaluation_example_wise_binned_common.hpp"
#include "rule_evaluation_example_wise_partial_fixed_common.hpp"


namespace boosting {

    /**
     * Allows to calculate the predictions of partial rules that predict for a predefined number of labels, as well as
     * an overall quality score, based on the gradients and Hessians that are stored by a
     * `DenseExampleWiseStatisticVector` using L1 and L2 regularization. The labels are assigned to bins based on the
     * gradients and Hessians.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class DenseExampleWiseFixedPartialBinnedRuleEvaluation final :
            public AbstractExampleWiseBinnedRuleEvaluation<DenseExampleWiseStatisticVector, PartialIndexVector> {

        private:

            const T& labelIndices_;

            std::unique_ptr<PartialIndexVector> indexVectorPtr_;

            PriorityQueue priorityQueue_;

        protected:

            uint32 calculateLabelWiseCriteria(const DenseExampleWiseStatisticVector& statisticVector, float64* criteria,
                                              uint32 numCriteria, float64 l1RegularizationWeight,
                                              float64 l2RegularizationWeight) override {
                uint32 numLabels = statisticVector.getNumElements();
                uint32 numPredictions = indexVectorPtr_->getNumElements();
                DenseExampleWiseStatisticVector::gradient_const_iterator gradientIterator =
                    statisticVector.gradients_cbegin();
                DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessianIterator =
                    statisticVector.hessians_diagonal_cbegin();
                typename T::const_iterator labelIndexIterator = labelIndices_.cbegin();
                sortLabelWiseCriteria(priorityQueue_, numPredictions, gradientIterator, hessianIterator,
                                      labelIndexIterator, numLabels, l1RegularizationWeight, l2RegularizationWeight);
                PartialIndexVector::iterator indexIterator = indexVectorPtr_->begin();

                for (uint32 i = 0; i < numCriteria; i++) {
                    const IndexedValue<float64>& entry = priorityQueue_.top();
                    uint32 index = entry.index;
                    indexIterator[i] = index;
                    criteria[i] = calculateLabelWiseScore(gradientIterator[index], hessianIterator[index],
                                                          l1RegularizationWeight, l2RegularizationWeight);
                    priorityQueue_.pop();
                }

                return numCriteria;
            }

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param maxBins                   The maximum number of bins
             * @param indexVectorPtr            An unique pointer to an object of type `PartialIndexVector` that stores
             *                                  the indices of the labels for which a rule predicts
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be
             *                                  used to assign labels to bins
             * @param blas                      A reference to an object of type `Blas` that allows to execute BLAS
             *                                  routines
             * @param lapack                    A reference to an object of type `Lapack` that allows to execute LAPACK
             *                                  routines
             */
            DenseExampleWiseFixedPartialBinnedRuleEvaluation(const T& labelIndices, uint32 maxBins,
                                                             std::unique_ptr<PartialIndexVector> indexVectorPtr,
                                                             float64 l1RegularizationWeight,
                                                             float64 l2RegularizationWeight,
                                                             std::unique_ptr<ILabelBinning> binningPtr,
                                                             const Blas& blas, const Lapack& lapack)
                : AbstractExampleWiseBinnedRuleEvaluation<DenseExampleWiseStatisticVector, PartialIndexVector>(
                      *indexVectorPtr, false, maxBins, l1RegularizationWeight, l2RegularizationWeight,
                      std::move(binningPtr), blas, lapack),
                  labelIndices_(labelIndices), indexVectorPtr_(std::move(indexVectorPtr)) {

            }

    };

    ExampleWiseFixedPartialBinnedRuleEvaluationFactory::ExampleWiseFixedPartialBinnedRuleEvaluationFactory(
            float32 labelRatio, uint32 minLabels, uint32 maxLabels, float64 l1RegularizationWeight,
            float64 l2RegularizationWeight, std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr,
            const Blas& blas, const Lapack& lapack)
        : labelRatio_(labelRatio), minLabels_(minLabels), maxLabels_(maxLabels),
          l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
          labelBinningFactoryPtr_(std::move(labelBinningFactoryPtr)), blas_(blas), lapack_(lapack) {

    }

    std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> ExampleWiseFixedPartialBinnedRuleEvaluationFactory::create(
            const DenseExampleWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        uint32 numPredictions = calculateBoundedFraction(statisticVector.getNumElements(), labelRatio_, minLabels_,
                                                         maxLabels_);
        std::unique_ptr<PartialIndexVector> indexVectorPtr = std::make_unique<PartialIndexVector>(numPredictions);
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        uint32 maxBins = labelBinningPtr->getMaxBins(numPredictions);
        return std::make_unique<DenseExampleWiseFixedPartialBinnedRuleEvaluation<CompleteIndexVector>>(
            indexVector, maxBins, std::move(indexVectorPtr), l1RegularizationWeight_, l2RegularizationWeight_,
            std::move(labelBinningPtr), blas_, lapack_);
    }

    std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> ExampleWiseFixedPartialBinnedRuleEvaluationFactory::create(
            const DenseExampleWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        uint32 maxBins = labelBinningPtr->getMaxBins(indexVector.getNumElements());
        return std::make_unique<DenseExampleWiseCompleteBinnedRuleEvaluation<PartialIndexVector>>(
            indexVector, maxBins, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr), blas_,
            lapack_);
    }

}
