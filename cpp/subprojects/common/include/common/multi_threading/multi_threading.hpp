/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/label_matrix.hpp"


/**
 * Defines an interface for all classes that allow to configure the multi-threading behavior of a parallelizable
 * algorithm.
 */
class IMultiThreadingConfig {

    public:

        virtual ~IMultiThreadingConfig() { };

        /**
         * Determines and returns the number of threads to be used by a parallelizable algorithm.
         *
         * @param labelMatrix   A reference to an object of type `ILabelMatrix` that provides access to the labels of
         *                      the training examples
         * @return              The number of threads to be used
         */
        virtual uint32 configure(const ILabelMatrix& labelMatrix) const = 0;

};
