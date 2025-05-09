/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

/**
 * An index and a corresponding value.
 *
 * @tparam T The type of the value
 */
template<typename T>
struct IndexedValue final {
    public:

        /**
         * Allows to compare two objects of type `IndexedValue` by their index.
         */
        struct CompareIndex final {
            public:

                /**
                 * Returns whether the a given object of type `IndexedValue` should go before a second one.
                 *
                 * @param lhs   A reference to a first object of type `IndexedValue`
                 * @param rhs   A reference to a second object of type `IndexedValue`
                 * @return      True, if the first object should go before the second one, false otherwise
                 */
                inline bool operator()(const IndexedValue<T>& lhs, const IndexedValue<T>& rhs) const {
                    return lhs.index < rhs.index;
                }
        };

        /**
         * Allows to compare two objects of type `IndexedValue` by their value.
         */
        struct CompareValue final {
            public:

                /**
                 * Returns whether the a given object of type `IndexedValue` should go before a second one.
                 *
                 * @param lhs   A reference to a first object of type `IndexedValue`
                 * @param rhs   A reference to a second object of type `IndexedValue`
                 * @return      True, if the first object should go before the second one, false otherwise
                 */
                inline bool operator()(const IndexedValue<T>& lhs, const IndexedValue<T>& rhs) const {
                    return lhs.value < rhs.value;
                }
        };

        IndexedValue() {}

        /**
         * @param index The index
         */
        IndexedValue(uint32 index) : index(index) {}

        /**
         * @param index The index
         * @param value The value
         */
        IndexedValue(uint32 index, T value) : index(index), value(value) {}

        /**
         * The index.
         */
        uint32 index;

        /**
         * The value.
         */
        T value;
};
