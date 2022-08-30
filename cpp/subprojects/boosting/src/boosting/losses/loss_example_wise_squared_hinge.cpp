#include "boosting/losses/loss_example_wise_squared_hinge.hpp"


namespace boosting {

    /**
     * An implementation of the type `IExampleWiseLoss` that implements a multi-label variant of the squared hinge loss
     * that is applied example-wise.
     */
    class ExampleWiseSquaredHingeLoss final : public IExampleWiseLoss {

        public:

            virtual void updateLabelWiseStatistics(uint32 exampleIndex,
                                                   const CContiguousConstView<const uint8>& labelMatrix,
                                                   const CContiguousConstView<float64>& scoreMatrix,
                                                   CompleteIndexVector::const_iterator labelIndicesBegin,
                                                   CompleteIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const override {
                // TODO Implement
            }

            virtual void updateLabelWiseStatistics(uint32 exampleIndex,
                                                   const CContiguousConstView<const uint8>& labelMatrix,
                                                   const CContiguousConstView<float64>& scoreMatrix,
                                                   PartialIndexVector::const_iterator labelIndicesBegin,
                                                   PartialIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const override {
                // TODO Implement
            }

            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                                   const CContiguousConstView<float64>& scoreMatrix,
                                                   CompleteIndexVector::const_iterator labelIndicesBegin,
                                                   CompleteIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const override {
                // TODO Implement
            }

            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                                   const CContiguousConstView<float64>& scoreMatrix,
                                                   PartialIndexVector::const_iterator labelIndicesBegin,
                                                   PartialIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const override {
                // TODO Implement
            }

            void updateExampleWiseStatistics(uint32 exampleIndex, const CContiguousConstView<const uint8>& labelMatrix,
                                             const CContiguousConstView<float64>& scoreMatrix,
                                             DenseExampleWiseStatisticView& statisticView) const override {
                // TODO Implement
            }

            void updateExampleWiseStatistics(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                             const CContiguousConstView<float64>& scoreMatrix,
                                             DenseExampleWiseStatisticView& statisticView) const override {
                // TODO Implement
            }

            /**
             * @see `IEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const CContiguousConstView<const uint8>& labelMatrix,
                             const CContiguousConstView<float64>& scoreMatrix) const override {
                // TODO Implement
                return 0;
            }

            /**
             * @see `IEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                             const CContiguousConstView<float64>& scoreMatrix) const override {
                // TODO Implement
                return 0;
            }

            /**
             * @see `IDistanceMeasure::measureDistance`
             */
            float64 measureDistance(const VectorConstView<uint32>& relevantLabelIndices,
                                    CContiguousView<float64>::value_const_iterator scoresBegin,
                                    CContiguousView<float64>::value_const_iterator scoresEnd) const override {
                // TODO Implement
                return 0;
            }

    };

    /**
     * Allows to create instances of the type `IExampleWiseLoss` that implement a multi-label variant of the squared
     * hinge loss that is applied example-wise.
     */
    class ExampleWiseSquaredHingeLossFactory final : public IExampleWiseLossFactory {

        public:

            std::unique_ptr<IExampleWiseLoss> createExampleWiseLoss() const override {
                return std::make_unique<ExampleWiseSquaredHingeLoss>();
            }

    };

    ExampleWiseSquaredHingeLossConfig::ExampleWiseSquaredHingeLossConfig(
            const std::unique_ptr<IHeadConfig>& headConfigPtr)
        : headConfigPtr_(headConfigPtr) {

    }

    std::unique_ptr<IStatisticsProviderFactory> ExampleWiseSquaredHingeLossConfig::createStatisticsProviderFactory(
            const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
            const Lapack& lapack, bool preferSparseStatistics) const {
        return headConfigPtr_->createStatisticsProviderFactory(featureMatrix, labelMatrix, *this, blas, lapack);
    }

    std::unique_ptr<IProbabilityFunctionFactory> ExampleWiseSquaredHingeLossConfig::createProbabilityFunctionFactory() const {
        return nullptr;
    }

    float64 ExampleWiseSquaredHingeLossConfig::getDefaultPrediction() const {
        return 0.5;
    }

    std::unique_ptr<IExampleWiseLossFactory> ExampleWiseSquaredHingeLossConfig::createExampleWiseLossFactory() const {
        return std::make_unique<ExampleWiseSquaredHingeLossFactory>();
    }

}