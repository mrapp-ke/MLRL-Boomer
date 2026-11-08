/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view.hpp"

#include <utility>
#include <vector>

/**
 * A view that provides access to a specific number of lists of type `std::vector`.
 *
 * @tparam T The type of the values, the view provides access to
 */
template<typename T>
class MLRLCOMMON_API IndexedListsView {
    public:

        /**
         * The number of lists in the view.
         */
        const uint32 numLists;

        /**
         * A pointer to an array that stores the lists the view provides access to.
         */
        std::vector<T>* array;

        /**
         * @param array     A pointer to an array of type `std::vector`, shape `(numLists)`, that stores lists
         * @param numLists  The number of lists in the view
         */
        IndexedListsView(std::vector<T>* array, uint32 numLists) : numLists(numLists), array(array) {}

        /**
         * @param other A const reference to an object of type `IndexedListsView` that should be copied
         */
        IndexedListsView(const IndexedListsView& other) : numLists(other.numLists), array(other.array) {}

        /**
         * @param other A reference to an object of type `IndexedListsView` that should be moved
         */
        IndexedListsView(IndexedListsView&& other) : numLists(other.numLists), array(other.array) {}

        virtual ~IndexedListsView() {}

        /**
         * The type of the values, the view provides access to.
         */
        using value_type = T;

        /**
         * An iterator that provides read-only access to the values in the view.
         */
        using value_const_iterator = View<T>::const_iterator;

        /**
         * An iterator that provides access to the values in the view and allows to modify them.
         */
        using value_iterator = View<T>::iterator;

        /**
         * Provides read-only access to an individual list in the view.
         */
        using const_list = const std::vector<T>&;

        /**
         * Provides access to an individual list in the view and allows to modify it.
         */
        using list = std::vector<T>&;

        /**
         * Returns a view that provides read-only access to a specific list in the view.
         *
         * @param listIndex The index of the list
         * @return          A `const_list`
         */
        const_list operator[](uint32 listIndex) const {
            return array[listIndex];
        }

        /**
         * Returns a view that provides access to a specific list in the view and allows to modify it.
         *
         * @param listIndex The index of the list
         * @return          A `list`
         */
        list operator[](uint32 listIndex) {
            return array[listIndex];
        }

        /**
         * Returns a `value_const_iterator` to the beginning of the values in a specific list.
         *
         * @param listIndex The index of the list
         * @return          A `value_const_iterator` to the beginning of the values
         */
        value_const_iterator values_cbegin(uint32 listIndex) const {
            return std::to_address(array[listIndex].cbegin());
        }

        /**
         * Returns a `value_const_iterator` to the end of the values in a specific list.
         *
         * @param listIndex The index of the list
         * @return          A `value_const_iterator` to the end of the values
         */
        value_const_iterator values_cend(uint32 listIndex) const {
            return std::to_address(array[listIndex].cend());
        }

        /**
         * Returns a `value_iterator` to the beginning of the values in a specific list.
         *
         * @param listIndex The index of the list
         * @return          A `value_iterator` to the beginning of the values
         */
        value_iterator values_begin(uint32 listIndex) {
            return std::to_address(array[listIndex].begin());
        }

        /**
         * Returns a `value_iterator` to the end of the values in a specific list.
         *
         * @param listIndex The index of the list
         * @return          A `value_iterator` to the end of the values
         */
        value_iterator values_end(uint32 listIndex) {
            return std::to_address(array[listIndex].end());
        }

        /**
         * Sets all values stored in the view to zero.
         */
        void clear() {
            for (uint32 i = 0; i < numLists; i++) {
                (*this)[i].clear();
            }
        }

        /**
         * Releases the ownership of the array that stores lists. As a result, the behavior of this view becomes
         * undefined and it should not be used anymore. The caller is responsible for freeing the memory that is
         * occupied by the array.
         *
         * @return A pointer to the array that stores lists
         */
        std::vector<T>* release() {
            std::vector<T>* ptr = array;
            array = nullptr;
            return ptr;
        }
};

/**
 * Allocates the memory for a view that provides access to a specific number of lists of type `std::vector`.
 *
 * @tparam View The type of the view
 */
template<typename View>
class MLRLCOMMON_API IndexedListsViewAllocator : public View {
    public:

        /**
         * @param numLists The number of lists in the view
         */
        explicit IndexedListsViewAllocator(uint32 numLists)
            : View(new std::vector<typename View::value_type>[numLists] {
              }, numLists) {}

        /**
         * @param other A reference to an object of type `IndexedListsViewAllocator` that should be copied
         */
        IndexedListsViewAllocator(const IndexedListsViewAllocator<View>& other) = delete;

        /**
         * @param other A reference to an object of type `IndexedListsViewAllocator` that should be moved
         */
        IndexedListsViewAllocator(IndexedListsViewAllocator<View>&& other) : View(std::move(other)) {
            other.release();
        }

        virtual ~IndexedListsViewAllocator() override {
            if (View::array) {
                delete[] View::array;
            }
        }
};

/**
 * Allocates the memory, a `IndexedListsView` provides access to.
 *
 * @tparam T The type of the values stored in the `IndexedListsView`
 */
template<typename T>
using AllocatedIndexedListsView = IndexedListsViewAllocator<IndexedListsView<T>>;
