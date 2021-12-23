/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/nominal_feature_mask.hpp"
#include <memory>


/**
 * Defines an interface for all classes that allow to check whether individual features are nominal or not in cases
 * where different types of features, i.e., nominal and numerical/ordinal ones, are available.
 */
class IMixedNominalFeatureMask : public INominalFeatureMask {

    public:

        virtual ~IMixedNominalFeatureMask() { };

        /**
         * Sets whether the feature at a specific index is nominal or not.
         *
         * @param featureIndex  The index of the feature
         * @param nominal       True, if the feature is nominal, false, if it is numerical/ordinal
         */
        virtual void setNominal(uint32 featureIndex, bool nominal) = 0;

};

/**
 * Allows to create instances of the type `IMixedNominalFeatureMask`.
 */
class MixedNominalFeatureMaskFactory final {

    private:

        uint32 numFeatures_;

    public:

        /**
         * @param numFeatures The total number of available features
         */
        MixedNominalFeatureMaskFactory(uint32 numFeatures);

        std::unique_ptr<IMixedNominalFeatureMask> create() const;

};
