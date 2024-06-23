/*
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_matrix.hpp"
#include "mlrl/common/input/feature_type.hpp"
#include "mlrl/common/input/output_matrix.hpp"

#include <memory>

/**
 * Defines an interface for methods that assign feature values to bins.
 */
class IFeatureBinning : public IFeatureType {
    public:

        virtual ~IFeatureBinning() override {}
};

/**
 * Defines an interface for all factories that allow to create instances of the type `IFeatureBinning`.
 */
class IFeatureBinningFactory {
    public:

        virtual ~IFeatureBinningFactory() {}

        /**
         * Creates and returns a new object of type `IFeatureBinning`.
         *
         * @return An unique pointer to an object of type `IFeatureBinning` that has been created or a null pointer, if
         *         no feature binning should be used
         */
        virtual std::unique_ptr<IFeatureBinning> create() const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a method that assigns feature values to bins.
 */
class IFeatureBinningConfig {
    public:

        virtual ~IFeatureBinningConfig() {}

        /**
         * Creates and returns a new object of type `IFeatureBinningFactory` according to the specified configuration.
         *
         * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the feature
         *                      values of the training examples
         * @param outputMatrix  A reference to an object of type `IOutputMatrix` that provides access to the ground
         *                      truth of the training examples
         * @return              An unique pointer to an object of type `IFeatureBinningFactory` that has been created
         */
        virtual std::unique_ptr<IFeatureBinningFactory> createFeatureBinningFactory(
          const IFeatureMatrix& featureMatrix, const IOutputMatrix& outputMatrix) const = 0;
};
