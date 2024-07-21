/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/util/blas.hpp"
#include "mlrl/boosting/util/lapack.hpp"
#include "mlrl/common/input/feature_matrix.hpp"
#include "mlrl/common/input/output_matrix.hpp"
#include "mlrl/common/statistics/statistics_provider.hpp"

#include <memory>

namespace boosting {

    /**
     * Returns whether a sparse representation of the gradients and Hessians should be preferred or not.
     *
     * @param outputMatrix      A reference to an object of type `IOutputMatrix` that provides row-wise access to the
     *                          ground truth of the training examples
     * @param defaultRuleUsed   True, if a default rule is used, false otherwise
     * @param partialHeadsUsed  True, if the partial heads are used by the rules, false otherwise
     * @return                  True, if a sparse representation should be preferred, false otherwise
     */
    static inline bool shouldSparseStatisticsBePreferred(const IOutputMatrix& outputMatrix, bool defaultRuleUsed,
                                                         bool partialHeadsUsed) {
        return outputMatrix.isSparse() && outputMatrix.getNumOutputs() > 120 && !defaultRuleUsed && partialHeadsUsed;
    }

    /**
     * Defines an interface for all classes that allow to configure which format should be used for storing statistics
     * about the quality of prediction for training examples.
     */
    class IStatisticsConfig {
        public:

            virtual ~IStatisticsConfig() {}

            /**
             * Creates and returns a new object of type `IClassificationStatisticsProviderFactory` according to the
             * specified configuration.
             *
             * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the
             *                      feature values of the training examples
             * @param labelMatrix   A reference to an object of type `IRowWiseLabelMatrix` that provides row-wise access
             *                      to the labels of the training examples
             * @param blas          A reference to an object of type `Blas` that allows to execute BLAS routines
             * @param lapack        A reference to an object of type `Lapack` that allows to execute LAPACK routines
             * @return              An unique pointer to an object of type `IClassificationStatisticsProviderFactory`
             *                      that has been created
             */
            virtual std::unique_ptr<IClassificationStatisticsProviderFactory>
              createClassificationStatisticsProviderFactory(const IFeatureMatrix& featureMatrix,
                                                            const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
                                                            const Lapack& lapack) const = 0;

            /**
             * Creates and returns a new object of type `IRegressionStatisticsProviderFactory` according to the
             * specified configuration.
             *
             * @param featureMatrix     A reference to an object of type `IFeatureMatrix` that provides access to the
             *                          feature values of the training examples
             * @param regressionMatrix  A reference to an object of type `IRowWiseRegressionMatrix` that provides
             *                          row-wise access to the regression scores of the training examples
             * @param blas              A reference to an object of type `Blas` that allows to execute BLAS routines
             * @param lapack            A reference to an object of type `Lapack` that allows to execute LAPACK routines
             * @return                  An unique pointer to an object of type `IRegressionStatisticsProviderFactory`
             *                          that has been created
             */
            virtual std::unique_ptr<IRegressionStatisticsProviderFactory> createRegressionStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix, const Blas& blas,
              const Lapack& lapack) const = 0;

            /**
             * Returns whether a dense format is used for storing statistics about the quality of predictions for
             * training examples or not.
             *
             * @return True, if a dense format is used, false otherwise
             */
            virtual bool isDense() const = 0;

            /**
             * Returns whether a sparse format is used for storing statistics about the quality of predictions for
             * training examples or not.
             *
             * @return True, if a sparse format is used, false otherwise
             */
            virtual bool isSparse() const = 0;
    };

};
