/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

/**
 * Defines an interface for all classes that represent the type of a feature.
 */
class IFeatureType {
    public:

        virtual ~IFeatureType() {};

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
};
