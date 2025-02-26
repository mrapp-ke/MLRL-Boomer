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
#include "mlrl/common/input/regression_matrix_row_wise.hpp"
#include "mlrl/common/statistics/statistics_provider.hpp"

#include <memory>

namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure the heads of the rules that should be induced by a
     * rule learner.
     */
    class IHeadConfig {
        public:

            /**
             * Provides access to the interface of an `IHeadConfig`, abstracting away certain configuration options that
             * have already been pre-determined.
             *
             * @tparam StatisticType The type that should be used for representing statistics
             */
            template<typename StatisticType>
            class IPreset {
                public:

                    virtual ~IPreset() {}

                    /**
                     * Creates and returns a new object of type `IClassificationStatisticsProviderFactory` according to
                     * the specified configuration.
                     *
                     * @param featureMatrix               A reference to an object of type `IFeatureMatrix` that
                     *                                    provides access to the feature values of the training examples
                     * @param labelMatrix                 A reference to an object of type `IRowWiseLabelMatrix` that
                     *                                    provides access to the labels of the training examples
                     * @param lossFactoryPtr              A reference to an unique pointer that stores an object of type
                     *                                    `IDecomposableClassificationLossFactory` that allows to create
                     *                                    the loss function that should be used
                     * @param evaluationMeasureFactoryPtr A reference to an unique pointer that stores an object of type
                     *                                    `IClassificationEvaluationMeasureFactory` that allows to
                     *                                    create the measure that should be used to assess the quality
                     *                                    of scores that are predicted for certain examples by comparing
                     *                                    them to the corresponding ground truth
                     * @return                            An unique pointer to an object of type
                     *                                    `IClassificationStatisticsProviderFactory` that has been
                     *                                    created
                     */
                    virtual std::unique_ptr<IClassificationStatisticsProviderFactory>
                      createClassificationStatisticsProviderFactory(
                        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
                        std::unique_ptr<IDecomposableClassificationLossFactory<StatisticType>>& lossFactoryPtr,
                        std::unique_ptr<IClassificationEvaluationMeasureFactory<StatisticType>>&
                          evaluationMeasureFactoryPtr) const = 0;

                    /**
                     * Creates and returns a new object of type `IClassificationStatisticsProviderFactory` according to
                     * the specified configuration.
                     *
                     * @param featureMatrix               A reference to an object of type `IFeatureMatrix` that
                     *                                    provides access to the feature values of the training examples
                     * @param labelMatrix                 A reference to an object of type `IRowWiseLabelMatrix` that
                     *                                    provides access to the labels of the training examples
                     * @param lossFactoryPtr              A reference to an unique pointer that stores an object of type
                     *                                    `ISparseDecomposableClassificationLossFactory` that allows to
                     *                                    create the loss function that should be used
                     * @param evaluationMeasureFactoryPtr A reference to an unique pointer that stores an object of type
                     *                                    `IClassificationEvaluationMeasureFactory` that allows to
                     *                                    create the measure that should be used to assess the quality
                     *                                    of scores that are predicted for certain examples by comparing
                     *                                    them to the corresponding ground truth
                     * @return                            An unique pointer to an object of type
                     *                                    `IClassificationStatisticsProviderFactory` that has been
                     *                                    created
                     */
                    virtual std::unique_ptr<IClassificationStatisticsProviderFactory>
                      createClassificationStatisticsProviderFactory(
                        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
                        std::unique_ptr<ISparseDecomposableClassificationLossFactory<StatisticType>>& lossFactoryPtr,
                        std::unique_ptr<ISparseEvaluationMeasureFactory<StatisticType>>& evaluationMeasureFactoryPtr)
                        const = 0;

                    /**
                     * Creates and returns a new object of type `IClassificationStatisticsProviderFactory` according to
                     * the specified configuration.
                     *
                     * @param featureMatrix               A reference to an object of type `IFeatureMatrix` that
                     *                                    provides access to the feature values of the training examples
                     * @param labelMatrix                 A reference to an object of type `IRowWiseLabelMatrix` that
                     *                                    provides access to the labels of the training examples
                     * @param lossFactoryPtr              A reference to an unique pointer that stores an object of type
                     *                                    `INonDecomposableClassificationLossFactory` that allows to
                     *                                    create the loss function that should be used
                     * @param evaluationMeasureFactoryPtr A reference to an unique pointer that stores an object of type
                     *                                    `IClassificationEvaluationMeasureFactory` that allows to
                     *                                    create the measure that should be used to assess the quality
                     *                                    of scores that are predicted for certain examples by comparing
                     *                                    them to the corresponding ground truth
                     * @param blasFactory                 A reference to an object of type `BlasFactory` that allows to
                     *                                    create objects for executing BLAS routines
                     * @param lapackFactory               A reference to an object of type `LapackFactory` that allows
                     *                                    to create objects for executing LAPACK routines
                     * @return                            An unique pointer to an object of type
                     *                                    `IClassificationStatisticsProviderFactory` that has been
                     *                                    created
                     */
                    virtual std::unique_ptr<IClassificationStatisticsProviderFactory>
                      createClassificationStatisticsProviderFactory(
                        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
                        std::unique_ptr<INonDecomposableClassificationLossFactory<StatisticType>>& lossFactoryPtr,
                        std::unique_ptr<IClassificationEvaluationMeasureFactory<StatisticType>>&
                          evaluationMeasureFactoryPtr,
                        const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const = 0;

                    /**
                     * Creates and returns a new object of type `IRegressionStatisticsProviderFactory` according to the
                     * specified configuration.
                     *
                     * @param featureMatrix               A reference to an object of type `IFeatureMatrix` that
                     *                                    provides access to the feature values of the training examples
                     * @param regressionMatrix            A reference to an object of type `IRowWiseLabelMatrix` that
                     *                                    provides access to the labels of the training examples
                     * @param lossFactoryPtr              A reference to an unique pointer that stores an object of type
                     *                                    `IDecomposableRegressionLossFactory` that allows to create the
                     *                                    loss function that should be used
                     * @param evaluationMeasureFactoryPtr A reference to an unique pointer that stores an object of type
                     *                                    `IRegressionEvaluationMeasureFactory` that allows to create
                     *                                    the measure that should be used to assess the quality of
                     *                                    scores that are predicted for certain examples by comparing
                     *                                    them to the corresponding ground truth
                     * @return                            An unique pointer to an object of type
                     *                                    `IRegressionStatisticsProviderFactory` that has been created
                     */
                    virtual std::unique_ptr<IRegressionStatisticsProviderFactory>
                      createRegressionStatisticsProviderFactory(
                        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
                        std::unique_ptr<IDecomposableRegressionLossFactory<StatisticType>>& lossFactoryPtr,
                        std::unique_ptr<IRegressionEvaluationMeasureFactory<StatisticType>>&
                          evaluationMeasureFactoryPtr) const = 0;

                    /**
                     * Creates and returns a new object of type `IRegressionStatisticsProviderFactory` according to the
                     * specified configuration.
                     *
                     * @param featureMatrix               A reference to an object of type `IFeatureMatrix` that
                     *                                    provides access to the feature values of the training examples
                     * @param regressionMatrix            A reference to an object of type `IRowWiseLabelMatrix` that
                     *                                    provides access to the labels of the training examples
                     * @param lossFactoryPtr              A reference to an unique pointer that stores an object of type
                     *                                    `INonDecomposableRegressionLossFactory` that allows to create
                     *                                    the loss function that should be used
                     * @param evaluationMeasureFactoryPtr A reference to an unique pointer that stores an object of type
                     *                                    `IRegressionEvaluationMeasureFactory` that allows to create
                     *                                    the measure that should be used to assess the quality of
                     *                                    scores that are predicted for certain examples by comparing
                     *                                    them to the corresponding ground truth
                     * @param blasFactory                 A reference to an object of type `BlasFactory` that allows to
                     *                                    create objects for executing BLAS routines
                     * @param lapackFactory               A reference to an object of type `LapackFactory` that allows
                     *                                    to create objects for executing LAPACK routines
                     * @return                            An unique pointer to an object of type
                     *                                    `IRegressionStatisticsProviderFactory` that has been created
                     */
                    virtual std::unique_ptr<IRegressionStatisticsProviderFactory>
                      createRegressionStatisticsProviderFactory(
                        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
                        std::unique_ptr<INonDecomposableRegressionLossFactory<StatisticType>>& lossFactoryPtr,
                        std::unique_ptr<IRegressionEvaluationMeasureFactory<StatisticType>>&
                          evaluationMeasureFactoryPtr,
                        const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const = 0;
            };

            virtual ~IHeadConfig() {}

            /**
             * Creates and returns a new object of type `IPreset<float32>`.
             *
             * @return An unique pointer to an object of type `IPreset<float32>` that has been created
             */
            virtual std::unique_ptr<IPreset<float32>> create32BitPreset() const = 0;

            /**
             * Creates and returns a new object of type `IPreset<float64>`.
             *
             * @return An unique pointer to an object of type `IPreset<float64>` that has been created
             */
            virtual std::unique_ptr<IPreset<float64>> create64BitPreset() const = 0;

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
