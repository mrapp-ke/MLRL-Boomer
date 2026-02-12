#include "mlrl/seco/statistics/statistics_provider_decomposable_dense.hpp"

#include "mlrl/common/math/vector_math.hpp"
#include "mlrl/common/simd/vector_math.hpp"
#include "mlrl/seco/data/vector_confusion_matrix_dense.hpp"
#include "statistics_decomposable_common.hpp"
#include "statistics_provider_decomposable.hpp"

#include <type_traits>

namespace seco {

    /**
     * Provides access to the elements of confusion matrices that are computed independently for each output and are
     * stored using dense data structures.
     *
     * @tparam LabelMatrix  The type of the matrix that provides access to the labels of the training examples
     * @tparam VectorMath   The type that implements basic operations for calculating with numerical arrays
     */
    template<typename LabelMatrix, typename VectorMath>
    class DenseDecomposableStatistics final
        : public AbstractDecomposableStatistics<DenseDecomposableStatisticMatrix<LabelMatrix>,
                                                IDecomposableRuleEvaluationFactory> {
        private:

            template<typename StatisticType>
            using StatisticVector = DenseConfusionMatrixVector<StatisticType, VectorMath>;

            template<typename WeightVector, typename IndexVector, typename StatisticType>
            using StatisticsSubset =
              CoverageStatisticsSubset<CoverageStatisticsState<DenseDecomposableStatisticMatrix<LabelMatrix>>,
                                       StatisticVector<StatisticType>, WeightVector, IndexVector,
                                       IDecomposableRuleEvaluationFactory>;

            template<typename WeightVector, typename StatisticType>
            using WeightedStatistics =
              WeightedStatistics<CoverageStatisticsState<DenseDecomposableStatisticMatrix<LabelMatrix>>,
                                 StatisticVector<StatisticType>, WeightVector, IDecomposableRuleEvaluationFactory>;

        public:

            /**
             * @param labelMatrix               A reference to an object of template type `LabelMatrix` that provides
             *                                  access to the labels of the training examples
             * @param coverageMatrixPtr         An unique pointer to an object of type `DenseCoverageMatrix` that stores
             *                                  how often individual examples and labels have been covered
             * @param majorityLabelVectorPtr    An unique pointer to an object of type `BinarySparseArrayVector` that
             *                                  stores the predictions of the default rule
             * @param ruleEvaluationFactory     A reference to an object of type `IDecomposableRuleEvaluationFactory`
             *                                  that allows to create instances of the class that is used for
             *                                  calculating the predictions of rules, as well as their overall quality
             */
            DenseDecomposableStatistics(const LabelMatrix& labelMatrix,
                                        std::unique_ptr<DenseCoverageMatrix> coverageMatrixPtr,
                                        std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr,
                                        const IDecomposableRuleEvaluationFactory& ruleEvaluationFactory)
                : AbstractDecomposableStatistics<DenseDecomposableStatisticMatrix<LabelMatrix>,
                                                 IDecomposableRuleEvaluationFactory>(
                    std::make_unique<DenseDecomposableStatisticMatrix<LabelMatrix>>(
                      labelMatrix, std::move(majorityLabelVectorPtr), std::move(coverageMatrixPtr)),
                    ruleEvaluationFactory) {}

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                            const EqualWeightVector& weights) const override {
                std::unique_ptr<StatisticVector<uint32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<StatisticsSubset<EqualWeightVector, CompleteIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const EqualWeightVector& weights) const override {
                std::unique_ptr<StatisticVector<uint32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<StatisticsSubset<EqualWeightVector, PartialIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                            const BitWeightVector& weights) const override {
                std::unique_ptr<StatisticVector<uint32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<StatisticsSubset<BitWeightVector, CompleteIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const BitWeightVector& weights) const override {
                std::unique_ptr<StatisticVector<uint32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<StatisticsSubset<BitWeightVector, PartialIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                            const DenseWeightVector<uint16>& weights) const override {
                std::unique_ptr<StatisticVector<uint32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<StatisticsSubset<DenseWeightVector<uint16>, CompleteIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const DenseWeightVector<uint16>& weights) const override {
                std::unique_ptr<StatisticVector<uint32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<StatisticsSubset<DenseWeightVector<uint16>, PartialIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                            const DenseWeightVector<float32>& weights) const override {
                std::unique_ptr<StatisticVector<float32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<float32>>(this->getNumOutputs(), true);
                return std::make_unique<StatisticsSubset<DenseWeightVector<float32>, CompleteIndexVector, float32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const DenseWeightVector<float32>& weights) const override {
                std::unique_ptr<StatisticVector<float32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<float32>>(this->getNumOutputs(), true);
                return std::make_unique<StatisticsSubset<DenseWeightVector<float32>, PartialIndexVector, float32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<EqualWeightVector>& weights) const override {
                std::unique_ptr<StatisticVector<uint32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<EqualWeightVector>, CompleteIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<EqualWeightVector>& weights) const override {
                std::unique_ptr<StatisticVector<uint32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<EqualWeightVector>, PartialIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<BitWeightVector>& weights) const override {
                std::unique_ptr<StatisticVector<uint32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<BitWeightVector>, CompleteIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<BitWeightVector>& weights) const override {
                std::unique_ptr<StatisticVector<uint32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<BitWeightVector>, PartialIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<uint16>>& weights) const override {
                std::unique_ptr<StatisticVector<uint32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<DenseWeightVector<uint16>>, CompleteIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<uint16>>& weights) const override {
                std::unique_ptr<StatisticVector<uint32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<DenseWeightVector<uint16>>, PartialIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<float32>>& weights) const override {
                std::unique_ptr<StatisticVector<uint32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<DenseWeightVector<float32>>, CompleteIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<float32>>& weights) const override {
                std::unique_ptr<StatisticVector<uint32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<DenseWeightVector<float32>>, PartialIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const EqualWeightVector& weights) const override {
                return std::make_unique<WeightedStatistics<EqualWeightVector, uint32>>(*this->statePtr_, weights,
                                                                                       *this->ruleEvaluationFactory_);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const BitWeightVector& weights) const override {
                return std::make_unique<WeightedStatistics<BitWeightVector, uint32>>(*this->statePtr_, weights,
                                                                                     *this->ruleEvaluationFactory_);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const DenseWeightVector<uint16>& weights) const override {
                return std::make_unique<WeightedStatistics<DenseWeightVector<uint16>, uint32>>(
                  *this->statePtr_, weights, *this->ruleEvaluationFactory_);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const DenseWeightVector<float32>& weights) const override {
                return std::make_unique<WeightedStatistics<DenseWeightVector<float32>, float32>>(
                  *this->statePtr_, weights, *this->ruleEvaluationFactory_);
            }
    };

    template<typename VectorMath>
    static inline std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> createStatistics(
      const IDecomposableRuleEvaluationFactory& ruleEvaluationFactory, const CContiguousView<const uint8>& labelMatrix,
      std::type_identity<VectorMath>) {
        uint32 numExamples = labelMatrix.numRows;
        uint32 numLabels = labelMatrix.numCols;
        std::unique_ptr<ResizableBinarySparseArrayVector> majorityLabelVectorPtr =
          std::make_unique<ResizableBinarySparseArrayVector>(numLabels);
        ResizableBinarySparseArrayVector::iterator majorityIterator = majorityLabelVectorPtr->begin();
        float64 threshold = numExamples / 2.0;
        float64 sumOfUncoveredWeights = 0;
        uint32 n = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 numRelevant = 0;

            for (uint32 j = 0; j < numExamples; j++) {
                uint8 trueLabel = labelMatrix.values_cbegin(j)[i];
                numRelevant += trueLabel;
            }

            if (numRelevant > threshold) {
                sumOfUncoveredWeights += (numExamples - numRelevant);
                majorityIterator[n] = i;
                n++;
            } else {
                sumOfUncoveredWeights += numRelevant;
            }
        }

        majorityLabelVectorPtr->setNumElements(n, true);
        std::unique_ptr<DenseCoverageMatrix> coverageMatrixPtr =
          std::make_unique<DenseCoverageMatrix>(numExamples, numLabels, sumOfUncoveredWeights);
        return std::make_unique<DenseDecomposableStatistics<CContiguousView<const uint8>, VectorMath>>(
          labelMatrix, std::move(coverageMatrixPtr),
          std::make_unique<BinarySparseArrayVector>(std::move(majorityLabelVectorPtr->getView())),
          ruleEvaluationFactory);
    }

    template<typename VectorMath>
    static inline std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> createStatistics(
      const IDecomposableRuleEvaluationFactory& ruleEvaluationFactory, const BinaryCsrView& labelMatrix,
      std::type_identity<VectorMath> vectorMath) {
        uint32 numExamples = labelMatrix.numRows;
        uint32 numLabels = labelMatrix.numCols;
        std::unique_ptr<ResizableBinarySparseArrayVector> majorityLabelVectorPtr =
          std::make_unique<ResizableBinarySparseArrayVector>(numLabels, true);
        ResizableBinarySparseArrayVector::iterator majorityIterator = majorityLabelVectorPtr->begin();

        for (uint32 i = 0; i < numExamples; i++) {
            BinaryCsrView::index_const_iterator indexIterator = labelMatrix.indices_cbegin(i);
            uint32 numElements = labelMatrix.indices_cend(i) - indexIterator;

            for (uint32 j = 0; j < numElements; j++) {
                uint32 index = indexIterator[j];
                majorityIterator[index] += 1;
            }
        }

        float64 threshold = numExamples / 2.0;
        float64 sumOfUncoveredWeights = 0;
        uint32 n = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 numRelevant = majorityIterator[i];

            if (numRelevant > threshold) {
                sumOfUncoveredWeights += (numExamples - numRelevant);
                majorityIterator[n] = i;
                n++;
            } else {
                sumOfUncoveredWeights += numRelevant;
            }
        }

        majorityLabelVectorPtr->setNumElements(n, true);
        std::unique_ptr<DenseCoverageMatrix> coverageMatrixPtr =
          std::make_unique<DenseCoverageMatrix>(numExamples, numLabels, sumOfUncoveredWeights);
        return std::make_unique<DenseDecomposableStatistics<BinaryCsrView, VectorMath>>(
          labelMatrix, std::move(coverageMatrixPtr),
          std::make_unique<BinarySparseArrayVector>(std::move(majorityLabelVectorPtr->getView())),
          ruleEvaluationFactory);
    }

    template<typename VectorMath>
    DenseDecomposableStatisticsProviderFactory<VectorMath>::DenseDecomposableStatisticsProviderFactory(
      std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
      std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
      std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr)
        : defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)) {}

    template<typename VectorMath>
    std::unique_ptr<IStatisticsProvider> DenseDecomposableStatisticsProviderFactory<VectorMath>::create(
      const CContiguousView<const uint8>& labelMatrix) const {
        std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> statisticsPtr =
          createStatistics(*defaultRuleEvaluationFactoryPtr_, labelMatrix, std::type_identity<VectorMath> {});
        return std::make_unique<DecomposableStatisticsProvider<IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    template<typename VectorMath>
    std::unique_ptr<IStatisticsProvider> DenseDecomposableStatisticsProviderFactory<VectorMath>::create(
      const BinaryCsrView& labelMatrix) const {
        std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> statisticsPtr =
          createStatistics(*defaultRuleEvaluationFactoryPtr_, labelMatrix, std::type_identity<VectorMath> {});
        return std::make_unique<DecomposableStatisticsProvider<IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    template class DenseDecomposableStatisticsProviderFactory<SequentialVectorMath>;

#if SIMD_SUPPORT_ENABLED
    template class DenseDecomposableStatisticsProviderFactory<SimdVectorMath>;
#endif
}
