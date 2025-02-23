/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/binning/label_binning.hpp"
#include "mlrl/boosting/rule_evaluation/head_type.hpp"
#include "mlrl/boosting/rule_evaluation/regularization.hpp"
#include "mlrl/common/multi_threading/multi_threading.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to configure single-output heads that predict for a single output.
     */
    class SingleOutputHeadConfig final : public IHeadConfig {
        private:

            const ReadableProperty<ILabelBinningConfig> labelBinningConfig_;

            const ReadableProperty<IMultiThreadingConfig> multiThreadingConfig_;

            const ReadableProperty<IRegularizationConfig> l1RegularizationConfig_;

            const ReadableProperty<IRegularizationConfig> l2RegularizationConfig_;

        public:

            /**
             * @param labelBinningConfig      A `ReadableProperty` that allows to access the `ILabelBinningConfig` that
             *                                stores the configuration of the method for assigning labels to bins
             * @param multiThreadingConfig    A `ReadableProperty` that allows to access the `IMultiThreadingConfig`
             *                                that stores the configuration of the multi-threading behavior that should
             *                                be used for the parallel update of statistics
             * @param l1RegularizationConfig  A `ReadableProperty` that allows to access the `IRegularizationConfig`
             *                                that stores the configuration of the L1 regularization
             * @param l2RegularizationConfig  A `ReadableProperty` that allows to access the `IRegularizationConfig`
             *                                that stores the configuration of the L2 regularization
             */
            SingleOutputHeadConfig(ReadableProperty<ILabelBinningConfig> labelBinningConfig,
                                   ReadableProperty<IMultiThreadingConfig> multiThreadingConfig,
                                   ReadableProperty<IRegularizationConfig> l1RegularizationConfig,
                                   ReadableProperty<IRegularizationConfig> l2RegularizationConfig);

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              std::unique_ptr<IDecomposableClassificationLossFactory<float64>>& lossFactoryPtr,
              std::unique_ptr<IClassificationEvaluationMeasureFactory<float64>>& evaluationMeasureFactoryPtr)
              const override;

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              std::unique_ptr<ISparseDecomposableClassificationLossFactory<float64>>& lossFactoryPtr,
              std::unique_ptr<ISparseEvaluationMeasureFactory<float64>>& evaluationMeasureFactoryPtr) const override;

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              std::unique_ptr<INonDecomposableClassificationLossFactory<float64>>& lossFactoryPtr,
              std::unique_ptr<IClassificationEvaluationMeasureFactory<float64>>& evaluationMeasureFactoryPtr,
              const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const override;

            std::unique_ptr<IRegressionStatisticsProviderFactory> createRegressionStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
              std::unique_ptr<IDecomposableRegressionLossFactory<float64>>& lossFactoryPtr,
              std::unique_ptr<IRegressionEvaluationMeasureFactory<float64>>& evaluationMeasureFactoryPtr)
              const override;

            std::unique_ptr<IRegressionStatisticsProviderFactory> createRegressionStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
              std::unique_ptr<INonDecomposableRegressionLossFactory<float64>>& lossFactoryPtr,
              std::unique_ptr<IRegressionEvaluationMeasureFactory<float64>>& evaluationMeasureFactoryPtr,
              const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const override;

            bool isPartial() const override;

            bool isSingleOutput() const override;
    };

}
