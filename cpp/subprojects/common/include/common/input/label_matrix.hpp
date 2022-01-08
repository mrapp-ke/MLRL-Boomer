/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_two_dimensional.hpp"


/**
 * Defines an interface for all label matrices.
 */
class ILabelMatrix : virtual public ITwoDimensionalView {

    public:

        virtual ~ILabelMatrix() override { };

};
