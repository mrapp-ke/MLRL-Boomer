/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/nominal_feature_mask.hpp"
#include <memory>


/**
 * Defines an interface for all classes that allow to check whether individual features are nominal or not in cases
 * where all features are of the same type, i.e., where all features are either nominal or numerical/ordinal.
 */
class IEqualNominalFeatureMask : public INominalFeatureMask {

    public:

        virtual ~IEqualNominalFeatureMask() { };

};

/**
 * Allows to create instances of the type `IEqualNominalFeatureMask`.
 */
class EqualNominalFeatureMaskFactory final {

    private:

        bool nominal_;

    public:

        /**
         * @param nominal True, if all features are nominal, false, if all features are numerical/ordinal
         */
        EqualNominalFeatureMaskFactory(bool nominal);

        std::unique_ptr<IEqualNominalFeatureMask> create() const;

};
