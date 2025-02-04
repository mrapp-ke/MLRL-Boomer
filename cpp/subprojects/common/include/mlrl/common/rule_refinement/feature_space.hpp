/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_info.hpp"
#include "mlrl/common/input/feature_matrix_column_wise.hpp"
#include "mlrl/common/rule_refinement/feature_subspace.hpp"
#include "mlrl/common/sampling/weight_vector_bit.hpp"
#include "mlrl/common/sampling/weight_vector_dense.hpp"
#include "mlrl/common/sampling/weight_vector_equal.hpp"
#include "mlrl/common/statistics/statistics_provider.hpp"

#include <memory>

/**
 * Defines an interface for all classes that provide access to the feature space.
 */
class IFeatureSpace {
    public:

        virtual ~IFeatureSpace() {}

        /**
         * Creates and returns a new subspace of this feature space.
         *
         * @param weights   A reference to an object of type `EqualWeightVector` that provides access to the weights of
         *                  individual training examples
         * @return          An unique pointer to an object of type `IFeatureSubspace` that has been created
         */
        virtual std::unique_ptr<IFeatureSubspace> createSubspace(const EqualWeightVector& weights) = 0;

        /**
         * Creates and returns a new subspace of this feature space.
         *
         * @param weights   A reference to an object of type `BitWeightVector` that provides access to the weights of
         *                  individual training examples
         * @return          An unique pointer to an object of type `IFeatureSubspace` that has been created
         */
        virtual std::unique_ptr<IFeatureSubspace> createSubspace(const BitWeightVector& weights) = 0;

        /**
         * Creates and returns a new subspace of this feature space.
         *
         * @param weights   A reference to an object of type `DenseWeightVector<uint32>` that provides access to the
         *                  weights of individual training examples
         * @return          An unique pointer to an object of type `IFeatureSubspace` that has been created
         */
        virtual std::unique_ptr<IFeatureSubspace> createSubspace(const DenseWeightVector<uint32>& weights) = 0;

        /**
         * Creates and returns a new subspace of this feature space.
         *
         * @param weights   A reference to an object of type `DenseWeightVector<float32>` that provides access to the
         *                  weights of individual training examples
         * @return          An unique pointer to an object of type `IFeatureSubspace` that has been created
         */
        virtual std::unique_ptr<IFeatureSubspace> createSubspace(const DenseWeightVector<float32>& weights) = 0;

        /**
         * Returns a reference to an object of type `IStatisticsProvider` that provides access to the statistics that
         * correspond to individual training examples in the feature space.
         *
         * @return A reference to an object of type `IStatisticsProvider`
         */
        virtual IStatisticsProvider& getStatisticsProvider() const = 0;
};

/**
 * Defines an interface for all classes that allow to create instances of the type `IFeatureSpace`.
 */
class IFeatureSpaceFactory {
    public:

        virtual ~IFeatureSpaceFactory() {}

        /**
         * Creates and returns a new object of type `IFeatureSpace`.
         *
         * @param featureMatrix         A reference to an object of type `IColumnWiseFeatureMatrix` that provides
         *                              column-wise access to the feature values of individual training examples
         * @param featureInfo           A reference  to an object of type `IFeatureInfo` that provides information about
         *                              the types of individual features
         * @param statisticsProvider    A reference to an object of type `IStatisticsProvider` that provides access to
         *                              statistics about the quality of predictions for training examples
         * @return                      An unique pointer to an object of type `IFeatureSpace` that has been created
         */
        virtual std::unique_ptr<IFeatureSpace> create(const IColumnWiseFeatureMatrix& featureMatrix,
                                                      const IFeatureInfo& featureInfo,
                                                      IStatisticsProvider& statisticsProvider) const = 0;
};
