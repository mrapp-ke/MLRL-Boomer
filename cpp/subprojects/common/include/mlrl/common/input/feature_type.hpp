/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_csc.hpp"
#include "mlrl/common/data/view_fortran_contiguous.hpp"
#include "mlrl/common/input/feature_vector.hpp"

/**
 * Defines an interface for all classes that represent the type of a feature.
 */
class IFeatureType {
    public:

        virtual ~IFeatureType() {}

        /**
         * Returns whether the feature is ordinal or not.
         *
         * @return True, if the feature is ordinal, false otherwise
         */
        // TODO Remove
        virtual bool isOrdinal() const = 0;

        /**
         * Returns whether the feature is nominal or not.
         *
         * @return True, if the feature is nominal, false otherwise
         */
        // TODO Remove
        virtual bool isNominal() const = 0;

        /**
         * Creates and returns a feature vector that stores the feature values taken from a given Fortran-contiguous
         * matrix for a certain feature.
         *
         * @param featureIndex  The index of the feature
         * @param featureMatrix A reference to an object of type `FortranContiguousConstView` that provides column-wise
         *                      access to the feature values
         * @return              An unique pointer to an object of type `IFeatureVector` that has been created
         */
        virtual std::unique_ptr<IFeatureVector> createFeatureVector(
          uint32 featureIndex, const FortranContiguousConstView<const float32>& featureMatrix) const = 0;

        /**
         * Creates and returns a feature vector that stores the feature values taken from a given CSC matrix for a
         * certain feature.
         *
         * @param featureIndex  The index of the feature
         * @param featureMatrix A reference to an object of type `CscConstView` that provides column-wise access to the
         *                      feature values
         * @return              An unique pointer to an object of type `IFeatureVector` that has been created
         */
        virtual std::unique_ptr<IFeatureVector> createFeatureVector(
          uint32 featureIndex, const CscConstView<const float32>& featureMatrix) const = 0;
};
