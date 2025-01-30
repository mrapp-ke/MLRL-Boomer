/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/util/blas.hpp"
#include "mlrl/boosting/util/lapack.hpp"
#include "mlrl/common/input/feature_matrix.hpp"
#include "mlrl/common/input/label_matrix_row_wise.hpp"
#include "mlrl/common/input/regression_matrix_row_wise.hpp"
#include "mlrl/common/statistics/statistics_provider.hpp"

#include <memory>

namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure which data type should be used for representing
     * statistics about the quality of predictions for training examples.
     */
    class IStatisticTypeConfig {
        public:

            virtual ~IStatisticTypeConfig() {}

            /**
             * Creates and returns a new object of type `IClassificationStatisticsProviderFactory` according to the
             * specified configuration.
             *
             * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the
             *                      feature values of the training examples
             * @param labelMatrix   A reference to an object of type `IRowWiseLabelMatrix` that provides access to the
             *                      labels of the training examples
             * @param blasFactory   A reference to an object of type `BlasFactory` that allows to create objects for
             *                      executing BLAS routines
             * @param lapackFactory A reference to an object of type `LapackFactory` that allows to create objects for
             *                      executing LAPACK routines
             * @return              An unique pointer to an object of type `IClassificationStatisticsProviderFactory`
             *                      that has been created
             */
            virtual std::unique_ptr<IClassificationStatisticsProviderFactory>
              createClassificationStatisticsProviderFactory(const IFeatureMatrix& featureMatrix,
                                                            const IRowWiseLabelMatrix& labelMatrix,
                                                            const BlasFactory& blasFactory,
                                                            const LapackFactory& lapackFactory) const = 0;

            /**
             * Creates and returns a new object of type `IRegressionStatisticsProviderFactory` according to the
             * specified configuration.
             *
             * @param featureMatrix     A reference to an object of type `IFeatureMatrix` that provides access to the
             *                          feature values of the training examples
             * @param regressionMatrix  A reference to an object of type `IRowWiseLabelMatrix` that provides access to
             *                          the labels of the training examples
             * @param blasFactory       A reference to an object of type `BlasFactory` that allows to create objects for
             *                          executing BLAS routines
             * @param lapackFactory     A reference to an object of type `LapackFactory` that allows to create objects
             *                          for executing LAPACK routines
             * @return                  An unique pointer to an object of type `IRegressionStatisticsProviderFactory`
             *                          that has been created
             */
            virtual std::unique_ptr<IRegressionStatisticsProviderFactory> createRegressionStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
              const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const = 0;
    };
}
