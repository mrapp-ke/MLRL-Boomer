/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once


/**
 * Defines an interface for all classes that represent the type of a feature.
 */
class IFeatureType {

    public:

        virtual ~IFeatureType() { };

        /**
         * Returns whether the feature is numerical/ordinal or not.
         *
         * @return True, if the feature is numerical/ordinal, false otherwise
         */
        virtual bool isNumerical() const = 0;

};
