/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

#include <memory>

// Forward declarations
class IRuleRefinement;
class IFeatureSubspace;

/**
 * Defines an interface for all classes that provide random access to indices.
 */
class IIndexVector {
    public:

        virtual ~IIndexVector() {}

        /**
         * Returns the number of indices.
         *
         * @return The number of indices
         */
        virtual uint32 getNumElements() const = 0;

        /**
         * Returns whether the indices are partial, i.e., some indices in the range [0, getNumElements()) are missing,
         * or not.
         *
         * @return True, if the indices are partial, false otherwise
         */
        virtual bool isPartial() const = 0;

        /**
         * Returns the index at a specific position.
         *
         * @param pos   The position of the index. Must be in [0, getNumElements())
         * @return      The index at the given position
         */
        virtual uint32 getIndex(uint32 pos) const = 0;

        /**
         * Creates and return a new instance of type `IRuleRefinement` that allows to search for the best refinement of
         * an existing rule that predicts only for the outputs whose indices are stored in this vector.
         *
         * @param featureSubspace   A reference to an object of type `IFeatureSubspace` that should be to search for the
         *                          refinement
         * @param featureIndex      The index of the feature that should be considered when searching for the refinement
         * @return                  An unique pointer to an object of type `IRuleRefinement` that has been created
         */
        virtual std::unique_ptr<IRuleRefinement> createRuleRefinement(IFeatureSubspace& featureSubspace,
                                                                      uint32 featureIndex) const = 0;
};
