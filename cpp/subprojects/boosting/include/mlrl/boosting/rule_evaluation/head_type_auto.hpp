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
     * Allows to configure a method that automatically decides for the type of rule heads to be used.
     */
    class AutomaticHeadConfig final : public IHeadConfig {
        private:

            const ReadableProperty<ILossConfig> lossConfig_;

            const ReadableProperty<ILabelBinningConfig> labelBinningConfig_;

            const ReadableProperty<IMultiThreadingConfig> multiThreadingConfig_;

            const ReadableProperty<IRegularizationConfig> l1RegularizationConfig_;

            const ReadableProperty<IRegularizationConfig> l2RegularizationConfig_;

        public:

            /**
             * @param lossConfigGetter              A `ReadableProperty` that allows to access the `ILossConfig` that
             *                                      stores the configuration of the loss function
             * @param labelBinningConfigGetter      A `ReadableProperty` that allows to access the `ILabelBinningConfig`
             *                                      that stores the configuration of the method for assigning labels to
             *                                      bins
             * @param multiThreadingConfigGetter    A `ReadableProperty` that allows to access the
             *                                      `IMultiThreadingConfig` that stores the configuration of the
             *                                      multi-threading behavior that should be used for the parallel update
             *                                      of statistics
             * @param l1RegularizationConfigGetter  A `ReadableProperty` that allows to access the
             *                                      `IRegularizationConfig` that stores the configuration of the L1
             *                                      regularization
             * @param l2RegularizationConfigGetter  A `ReadableProperty` that allows to access the
             *                                      `IRegularizationConfig` that stores the configuration of the L2
             *                                      regularization
             */
            AutomaticHeadConfig(ReadableProperty<ILossConfig> lossConfigGetter,
                                ReadableProperty<ILabelBinningConfig> labelBinningConfigGetter,
                                ReadableProperty<IMultiThreadingConfig> multiThreadingConfigGetter,
                                ReadableProperty<IRegularizationConfig> l1RegularizationConfigGetter,
                                ReadableProperty<IRegularizationConfig> l2RegularizationConfigGetter);

            std::unique_ptr<IClassificationStatisticsProviderFactory> createStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              const IDecomposableClassificationLossConfig& lossConfig) const override;

            std::unique_ptr<IClassificationStatisticsProviderFactory> createStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              const ISparseDecomposableClassificationLossConfig& lossConfig) const override;

            std::unique_ptr<IClassificationStatisticsProviderFactory> createStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              const INonDecomposableClassificationLossConfig& lossConfig, const Blas& blas,
              const Lapack& lapack) const override;

            std::unique_ptr<IRegressionStatisticsProviderFactory> createStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
              const IDecomposableRegressionLossConfig& lossConfig) const override;

            std::unique_ptr<IRegressionStatisticsProviderFactory> createStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
              const INonDecomposableRegressionLossConfig& lossConfig, const Blas& blas,
              const Lapack& lapack) const override;

            bool isPartial() const override;

            bool isSingleOutput() const override;
    };

}
