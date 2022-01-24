/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/feature_matrix.hpp"
#include "common/input/label_matrix.hpp"
#include "common/measures/measure_evaluation.hpp"
#include "common/measures/measure_similarity.hpp"
#include "common/statistics/statistics_provider.hpp"


namespace boosting {

    /**
     * Defines an interface for all loss functions.
     */
    class ILoss : public IEvaluationMeasure, public ISimilarityMeasure {

        public:

            virtual ~ILoss() override { };

    };

    /**
     * Defines an interface for all classes that allow to configure a loss function.
     */
    class ILossConfig {

        public:

            virtual ~ILossConfig() { };

            /**
             * Creates and returns a new object of type `IStatisticsProviderFactory` according to the specified
             * configuration.
             *
             * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the
             *                      feature values of the training examples
             * @param labelMatrix   A reference to an object of type `ILabelMatrix` that provides access to the labels
             *                      of the training examples
             * @return              An unique pointer to an object of type `IStatisticsProviderFactory` that has been
             *                      created
             */
            virtual std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
                const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix) const = 0;

    };

};
