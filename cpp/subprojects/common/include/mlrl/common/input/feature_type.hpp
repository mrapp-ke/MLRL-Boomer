/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_csc.hpp"
#include "mlrl/common/data/view_matrix_fortran_contiguous.hpp"
#include "mlrl/common/input/feature_vector.hpp"

/**
 * Defines an interface for all classes that represent the type of a feature.
 */
class IFeatureType {
    public:

        virtual ~IFeatureType() {}

        /**
         * Creates and returns a feature vector that stores the feature values taken from a given Fortran-contiguous
         * matrix for a certain feature.
         *
         * @param featureIndex  The index of the feature
         * @param featureMatrix A reference to an object of type `FortranContiguousView` that provides column-wise
         *                      access to the feature values
         * @return              An unique pointer to an object of type `IFeatureVector` that has been created
         */
        virtual std::unique_ptr<IFeatureVector> createFeatureVector(
          uint32 featureIndex, const FortranContiguousView<const float32>& featureMatrix) const = 0;

        /**
         * Creates and returns a feature vector that stores the feature values taken from a given CSC matrix for a
         * certain feature.
         *
         * @param featureIndex  The index of the feature
         * @param featureMatrix A reference to an object of type `CscView` that provides column-wise access to the
         *                      feature values
         * @return              An unique pointer to an object of type `IFeatureVector` that has been created
         */
        virtual std::unique_ptr<IFeatureVector> createFeatureVector(
          uint32 featureIndex, const CscView<const float32>& featureMatrix) const = 0;
};
