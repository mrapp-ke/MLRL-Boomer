/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/input/label_vector.hpp"
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
                uint32 numElements = v->getNumElements();
                LabelVector::index_const_iterator it = v->indices_cbegin();
                std::size_t hash = (std::size_t) numElements;

                for (uint32 i = 0; i < numElements; i++) {
                    hash ^= it[i] + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }

                return hash;
            }

        };

        /**
         * Allows to check whether two objects of type `LabelVector` are equal.
         */
        struct EqualsFunction {

            inline bool operator()(const std::unique_ptr<LabelVector>& lhs,
                                   const std::unique_ptr<LabelVector>& rhs) const {
                uint32 numElements = lhs->getNumElements();

                if (numElements != rhs->getNumElements()) {
                    return false;
                }

                LabelVector::index_const_iterator it1 = lhs->indices_cbegin();
                LabelVector::index_const_iterator it2 = rhs->indices_cbegin();

                for (uint32 i = 0; i < numElements; i++) {
                    if (it1[i] != it2[i]) {
                        return false;
                    }
                }

                return true;
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
