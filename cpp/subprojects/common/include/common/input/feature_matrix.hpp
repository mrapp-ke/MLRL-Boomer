/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_two_dimensional.hpp"


/**
 * Defines an interface for all feature matrices.
 */
class IFeatureMatrix : public ITwoDimensionalView {

    public:

        virtual ~IFeatureMatrix() override { };

};
