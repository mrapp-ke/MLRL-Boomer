/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss_decomposable_sparse.hpp"
#include "mlrl/boosting/losses/loss_non_decomposable.hpp"
#include "mlrl/boosting/util/blas.hpp"
#include "mlrl/boosting/util/lapack.hpp"
#include "mlrl/common/input/feature_matrix.hpp"
#include "mlrl/common/input/label_matrix_row_wise.hpp"

#include <memory>

namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure the heads of the rules that should be induced by a
     * rule learner.
     */
    class IHeadConfig {
        public:

            virtual ~IHeadConfig() {}

            /**
             * Creates and returns a new object of type `IClassificationStatisticsProviderFactory` according to the
             * specified configuration.
             *
             * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the
             *                      feature values of the training examples
             * @param labelMatrix   A reference to an object of type `IRowWiseLabelMatrix` that provides access to the
             *                      labels of the training examples
             * @param lossConfig    A reference to an object of type `IDecomposableClassificationLossConfig` that
             *                      specifies the configuration of the loss function
             * @return              An unique pointer to an object of type `IClassificationStatisticsProviderFactory`
             *                      that has been created
             */
            virtual std::unique_ptr<IClassificationStatisticsProviderFactory>
              createClassificationStatisticsProviderFactory(
                const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
                const IDecomposableClassificationLossConfig& lossConfig) const = 0;

            /**
             * Creates and returns a new object of type `IClassificationStatisticsProviderFactory` according to the
             * specified configuration.
             *
             * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the
             *                      feature values of the training examples
             * @param labelMatrix   A reference to an object of type `IRowWiseLabelMatrix` that provides access to the
             *                      labels of the training examples
             * @param lossConfig    A reference to an object of type `ISparseDecomposableClassificationLossConfig` that
             *                      specifies the configuration of the loss function
             * @return              An unique pointer to an object of type `IClassificationStatisticsProviderFactory`
             *                      that has been created
             */
            virtual std::unique_ptr<IClassificationStatisticsProviderFactory>
              createClassificationStatisticsProviderFactory(
                const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
                const ISparseDecomposableClassificationLossConfig& lossConfig) const = 0;

            /**
             * Creates and returns a new object of type `IClassificationStatisticsProviderFactory` according to the
             * specified configuration.
             *
             * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the
             *                      feature values of the training examples
             * @param labelMatrix   A reference to an object of type `IRowWiseLabelMatrix` that provides access to the
             *                      labels of the training examples
             * @param lossConfig    A reference to an object of type `INonDecomposableClassificationLossConfig` that
             *                      specifies the configuration of the loss function
             * @param blas          A reference to an object of type `Blas` that allows to execute BLAS routines
             * @param lapack        A reference to an object of type `Lapack` that allows to execute LAPACK routines
             * @return              An unique pointer to an object of type `IClassificationStatisticsProviderFactory`
             *                      that has been created
             */
            virtual std::unique_ptr<IClassificationStatisticsProviderFactory>
              createClassificationStatisticsProviderFactory(const IFeatureMatrix& featureMatrix,
                                                            const IRowWiseLabelMatrix& labelMatrix,
                                                            const INonDecomposableClassificationLossConfig& lossConfig,
                                                            const Blas& blas, const Lapack& lapack) const = 0;

            /**
             * Creates and returns a new object of type `IRegressionStatisticsProviderFactory` according to the
             * specified configuration.
             *
             * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the
             *                      feature values of the training examples
             * @param regressionMatrix  A reference to an object of type `IRowWiseLabelMatrix` that provides access to
             * the labels of the training examples
             * @param lossConfig    A reference to an object of type `IDecomposableRegressionLossConfig` that specifies
             *                      the configuration of the loss function
             * @return              An unique pointer to an object of type `IRegressionStatisticsProviderFactory` that
             *                      has been created
             */
            virtual std::unique_ptr<IRegressionStatisticsProviderFactory> createRegressionStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
              const IDecomposableRegressionLossConfig& lossConfig) const = 0;

            /**
             * Creates and returns a new object of type `IRegressionStatisticsProviderFactory` according to the
             * specified configuration.
             *
             * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the
             *                      feature values of the training examples
             * @param regressionMatrix  A reference to an object of type `IRowWiseLabelMatrix` that provides access to
             * the labels of the training examples
             * @param lossConfig    A reference to an object of type `INonDecomposableRegressionLossConfig` that
             *                      specifies the configuration of the loss function
             * @param blas          A reference to an object of type `Blas` that allows to execute BLAS routines
             * @param lapack        A reference to an object of type `Lapack` that allows to execute LAPACK routines
             * @return              An unique pointer to an object of type `IRegressionStatisticsProviderFactory` that
             *                      has been created
             */
            virtual std::unique_ptr<IRegressionStatisticsProviderFactory> createRegressionStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
              const INonDecomposableRegressionLossConfig& lossConfig, const Blas& blas, const Lapack& lapack) const = 0;

            /**
             * Returns, whether the heads of rules are partial, i.e., they predict for a subset of the available
             * outputs, or not.
             *
             * @return True, if the heads of rules are partial, false otherwise
             */
            virtual bool isPartial() const = 0;

            /**
             * Returns whether the rule heads predict for a single output or not.
             *
             * @return True, if the rule heads predict for a single output, false otherwise
             */
            virtual bool isSingleOutput() const = 0;
    };

}
