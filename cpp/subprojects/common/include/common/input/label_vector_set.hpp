/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/input/label_vector.hpp"
#include "common/data/functions.hpp"
#include <unordered_map>
#include <memory>


/**
 * An unordered set that allows to store unique label vectors, as well as values that are associated with them.
 *
 * @tparam T The type of the values that are associated with the label vectors
 */
template<class T>
class LabelVectorSet {

    private:

        /**
         * Allows to compute hashes for objects of type `LabelVector`.
         */
        struct HashFunction {

            inline std::size_t operator()(const std::unique_ptr<LabelVector>& v) const {
                return hashArray(v->indices_cbegin(), v->getNumElements());
            }

        };

        /**
         * Allows to check whether two objects of type `LabelVector` are equal.
         */
        struct EqualsFunction {

            inline bool operator()(const std::unique_ptr<LabelVector>& lhs,
                                   const std::unique_ptr<LabelVector>& rhs) const {
                return compareArrays(lhs->indices_cbegin(), lhs->getNumElements(), rhs->indices_cbegin(),
                                     rhs->getNumElements());
            }

        };

        typedef std::unordered_map<std::unique_ptr<LabelVector>, T, HashFunction, EqualsFunction> Map;

        Map map_;

    public:

        /**
         * An iterator that provides read-only access to the label vectors and their associated values.
         */
        typedef typename Map::const_iterator const_iterator;

        /**
         * Returns a `const_iterator` to the beginning of the label vectors and their associated values.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the label vectors and their associated values.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const;

        /**
         * Adds a new label vector to the set.
         *
         * @param labelVectorPtr    An unique pointer to an object of type `LabelVector`
         * @return                  A reference to an object of template type `T` that is associated with the given
         *                          label vector
         */
        T& addLabelVector(std::unique_ptr<LabelVector> labelVectorPtr);

};
