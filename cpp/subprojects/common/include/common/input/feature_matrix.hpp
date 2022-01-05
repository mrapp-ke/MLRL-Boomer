/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"


/**
 * Defines an interface for all feature matrices.
 */
class IFeatureMatrix {

    public:

        virtual ~IFeatureMatrix() { };

        /**
         * Returns the number of available examples.
         *
         * @return The number of examples
         */
        virtual uint32 getNumRows() const = 0;

        /**
         * Returns the number of available features.
         *
         * @return The number of features
         */
        virtual uint32 getNumCols() const = 0;

        /**
         * Returns whether the feature matrix is sparse or not.
         *
         * @return True, if the feature matrix is sparse, false otherwise
         */
        virtual bool isSparse() const = 0;

};
