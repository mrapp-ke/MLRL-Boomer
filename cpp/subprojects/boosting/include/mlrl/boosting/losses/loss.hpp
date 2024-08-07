/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/prediction/probability_function_joint.hpp"
#include "mlrl/boosting/prediction/probability_function_marginal.hpp"
#include "mlrl/boosting/util/blas.hpp"
#include "mlrl/boosting/util/lapack.hpp"
#include "mlrl/common/input/feature_matrix.hpp"
#include "mlrl/common/input/label_matrix_row_wise.hpp"
#include "mlrl/common/input/regression_matrix_row_wise.hpp"
#include "mlrl/common/measures/measure_distance.hpp"
#include "mlrl/common/measures/measure_evaluation.hpp"
#include "mlrl/common/statistics/statistics_provider.hpp"

#include <memory>

namespace boosting {

    /**
     * Defines an interface for all loss functions that can be used in classification problems.
     */
    class IClassificationLoss : public IDistanceMeasure {
        public:

            virtual ~IClassificationLoss() override {}
    };

    /**
     * Defines an interface for all loss functions that can be used in regression problems.
     */
    class IRegressionLoss {
        public:

            virtual ~IRegressionLoss() {}
    };

    /**
     * Defines an interface for all classes that allow to configure a loss function.
     */
    class ILossConfig {
        public:

            virtual ~ILossConfig() {}

            /**
             * Returns whether the loss function is decomposable or not.
             *
             * @return True, if the loss function is decomposable, false otherwise
             */
            virtual bool isDecomposable() const = 0;

            /**
             * Returns whether the loss function supports to use a sparse format for storing statistics or not.
             *
             * @return True, if the loss function supports to use a sparse format for storing statistics, false
             *         otherwise
             */
            virtual bool isSparse() const = 0;

            /**
             * Returns the default prediction for an example that is not covered by any rules.
             *
             * @return The default prediction
             */
            virtual float64 getDefaultPrediction() const = 0;
    };

    /**
     * Defines an interface for all classes that allow to configure a loss function that can be used in classification
     * problems.
     */
    class IClassificationLossConfig : virtual public ILossConfig {
        public:

            virtual ~IClassificationLossConfig() override {}

            /**
             * Creates and returns a new object of type `IClassificationStatisticsProviderFactory` according to the
             * specified configuration.
             *
             * @param featureMatrix             A reference to an object of type `IFeatureMatrix` that provides access
             *                                  to the feature values of the training examples
             * @param labelMatrix               A reference to an object of type `IRowWiseLabelMatrix` that provides
             *                                  access to the labels of the training examples
             * @param blas                      A reference to an object of type `Blas` that allows to execute BLAS
             *                                  routines
             * @param lapack                    A reference to an object of type `Lapack` that allows to execute LAPACK
             *                                  routines
             * @param preferSparseStatistics    True, if a sparse representation of statistics should be preferred, if
             *                                  possible, false otherwise
             * @return                          An unique pointer to an object of type
             *                                  `IClassificationStatisticsProviderFactory` that has been created
             */
            virtual std::unique_ptr<IClassificationStatisticsProviderFactory>
              createClassificationStatisticsProviderFactory(const IFeatureMatrix& featureMatrix,
                                                            const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
                                                            const Lapack& lapack,
                                                            bool preferSparseStatistics) const = 0;

            /**
             * Creates and returns a new object of type `IClassificationEvaluationMeasureFactory` according to the
             * specified configuration.
             *
             * @return An unique pointer to an object of type `IClassificationEvaluationMeasureFactory` that has been
             *         created
             */
            virtual std::unique_ptr<IClassificationEvaluationMeasureFactory>
              createClassificationEvaluationMeasureFactory() const = 0;

            /**
             * Creates and returns a new object of type `IDistanceMeasureFactory` according to the specified
             * configuration.
             *
             * @return An unique pointer to an object of type `IDistanceMeasureFactory` that has been created
             */
            virtual std::unique_ptr<IDistanceMeasureFactory> createDistanceMeasureFactory() const = 0;

            /**
             * Creates and returns a new object of type `IMarginalProbabilityFunctionFactory` according to the specified
             * configuration.
             *
             * @return An unique pointer to an object of type `IMarginalProbabilityFunctionFactory` that has been
             *         created or a null pointer, if the loss function does not support the prediction of marginal
             *         probabilities
             */
            virtual std::unique_ptr<IMarginalProbabilityFunctionFactory> createMarginalProbabilityFunctionFactory()
              const = 0;

            /**
             * Creates and returns a new object of type `IJointProbabilityFunctionFactory` according to the specified
             * configuration.
             *
             * @return An unique pointer to an object of type `IJointProbabilityFunctionFactory` that has been created
             *         to a null pointer, if the loss function does not support the prediction of joint probabilities
             */
            virtual std::unique_ptr<IJointProbabilityFunctionFactory> createJointProbabilityFunctionFactory() const = 0;
    };

    /**
     * Defines an interface for all classes that allow to configure a loss function that can be used in regression
     * problems.
     */
    class IRegressionLossConfig : virtual public ILossConfig {
        public:

            virtual ~IRegressionLossConfig() override {}

            /**
             * Creates and returns a new object of type `IRegressionStatisticsProviderFactory` according to the
             * specified configuration.
             *
             * @param featureMatrix             A reference to an object of type `IFeatureMatrix` that provides access
             *                                  to the feature values of the training examples
             * @param regressionMatrix          A reference to an object of type `IRowWiseRegressionMatrix` that
             *                                  provides access to the regression scores of the training examples
             * @param blas                      A reference to an object of type `Blas` that allows to execute BLAS
             *                                  routines
             * @param lapack                    A reference to an object of type `Lapack` that allows to execute LAPACK
             *                                  routines
             * @param preferSparseStatistics    True, if a sparse representation of statistics should be preferred, if
             *                                  possible, false otherwise
             * @return                          An unique pointer to an object of type
             *                                  `IRegressionStatisticsProviderFactory` that has been created
             */
            virtual std::unique_ptr<IRegressionStatisticsProviderFactory> createRegressionStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix, const Blas& blas,
              const Lapack& lapack, bool preferSparseStatistics) const = 0;

            /**
             * Creates and returns a new object of type `IRegressionEvaluationMeasureFactory` according to the specified
             * configuration.
             *
             * @return An unique pointer to an object of type `IRegressionEvaluationMeasureFactory` that has been
             *         created
             */
            virtual std::unique_ptr<IRegressionEvaluationMeasureFactory> createRegressionEvaluationMeasureFactory()
              const = 0;
    };

};
