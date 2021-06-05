/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/tuple.hpp"
#include <iterator>


namespace boosting {

    /**
     * An iterator that provides read-only access to the gradients that are stored in a C-contiguous array.
     */
    class DenseGradientConstIterator {

        private:

            const Tuple<float64>* ptr_;

        public:

            /**
             * @param ptr A pointer to an array of type `Tuple<float64>` that stores gradients and Hessians
             */
            DenseGradientConstIterator(const Tuple<float64>* ptr);

            /**
             * The type that is used to represent the difference between two iterators.
             */
            typedef int difference_type;

            /**
             * The type of the elements, the iterator provides access to.
             */
            typedef float64 value_type;

            /**
             * The type of a pointer to an element, the iterator provides access to.
             */
            typedef const float64* pointer;

            /**
             * The type of a reference to an element, the iterator provides access to.
             */
            typedef const float64& reference;

            /**
             * The tag that specifies the capabilities of the iterator.
             */
            typedef std::random_access_iterator_tag iterator_category;

            /**
             * Returns the element at a specific index.
             *
             * @param index The index of the element to be returned
             * @return      The element at the given index
             */
            reference operator[](uint32 index) const;

            /**
             * Returns the element, the iterator currently refers to.
             *
             * @return The element, the iterator currently refers to
             */
            reference operator*() const;

            /**
             * Returns an iterator to the next element.
             *
             * @return A reference to an iterator that refers to the next element
             */
            DenseGradientConstIterator& operator++();

            /**
             * Returns an iterator to the next element.
             *
             * @return A reference to an iterator that refers to the next element
             */
            DenseGradientConstIterator& operator++(int n);

            /**
             * Returns an iterator to the previous element.
             *
             * @return A reference to an iterator that refers to the previous element
             */
            DenseGradientConstIterator& operator--();

            /**
             * Returns an iterator to the previous element.
             *
             * @return A reference to an iterator that refers to the previous element
             */
            DenseGradientConstIterator& operator--(int n);

            /**
             * Returns whether this iterator and another one refer to the same element.
             *
             * @param rhs   A reference to another iterator
             * @return      True, if the iterators do not refer to the same element, false otherwise
             */
            bool operator!=(const DenseGradientConstIterator& rhs) const;

            /**
             * Returns whether this iterator and another one refer to the same element.
             *
             * @param rhs   A reference to another iterator
             * @return      True, if the iterators refer to the same element, false otherwise
             */
            bool operator==(const DenseGradientConstIterator& rhs) const;

            /**
             * Returns the difference between this iterator and another one.
             *
             * @param rhs   A reference to another iterator
             * @return      The difference between the iterators
             */
            difference_type operator-(const DenseGradientConstIterator& rhs) const;

    };

    /**
     * An iterator that provides access to the gradients that are stored in a C-contiguous array and allows to modify
     * them.
     */
    class DenseGradientIterator {

        private:

            Tuple<float64>* ptr_;

        public:

            /**
             * @param ptr A pointer to an array of type `Tuple<float64>` that stores gradients and Hessians
             */
            DenseGradientIterator(Tuple<float64>* ptr);

            /**
             * The type that is used to represent the difference between two iterators.
             */
            typedef int difference_type;

            /**
             * The type of the elements, the iterator provides access to.
             */
            typedef float64 value_type;

            /**
             * The type of a pointer to an element, the iterator provides access to.
             */
            typedef float64* pointer;

            /**
             * The type of a reference to an element, the iterator provides access to.
             */
            typedef float64& reference;

            /**
             * The tag that specifies the capabilities of the iterator.
             */
            typedef std::random_access_iterator_tag iterator_category;

            /**
             * Returns the element at a specific index.
             *
             * @param index The index of the element to be returned
             * @return      The element at the given index
             */
            reference operator[](uint32 index) const;

            /**
             * Returns the element, the iterator currently refers to.
             *
             * @return The element, the iterator currently refers to
             */
            reference operator*() const;

            /**
             * Returns an iterator to the next element.
             *
             * @return A reference to an iterator that refers to the next element
             */
            DenseGradientIterator& operator++();

            /**
             * Returns an iterator to the next element.
             *
             * @return A reference to an iterator that refers to the next element
             */
            DenseGradientIterator& operator++(int n);

            /**
             * Returns an iterator to the previous element.
             *
             * @return A reference to an iterator that refers to the previous element
             */
            DenseGradientIterator& operator--();

            /**
             * Returns an iterator to the previous element.
             *
             * @return A reference to an iterator that refers to the previous element
             */
            DenseGradientIterator& operator--(int n);

            /**
             * Returns whether this iterator and another one refer to the same element.
             *
             * @param rhs   A reference to another iterator
             * @return      True, if the iterators do not refer to the same element, false otherwise
             */
            bool operator!=(const DenseGradientIterator& rhs) const;

            /**
             * Returns whether this iterator and another one refer to the same element.
             *
             * @param rhs   A reference to another iterator
             * @return      True, if the iterators refer to the same element, false otherwise
             */
            bool operator==(const DenseGradientIterator& rhs) const;

            /**
             * Returns the difference between this iterator and another one.
             *
             * @param rhs   A reference to another iterator
             * @return      The difference between the iterators
             */
            difference_type operator-(const DenseGradientIterator& rhs) const;

    };

    /**
     * An iterator that provides read-only access to the Hessians that are stored in a C-contiguous array.
     */
    class DenseHessianConstIterator {

        private:

            const Tuple<float64>* ptr_;

        public:

            /**
             * @param ptr A pointer to an array of type `Tuple<float64>` that stores gradients and Hessians
             */
            DenseHessianConstIterator(const Tuple<float64>* ptr);

            /**
             * The type that is used to represent the difference between two iterators.
             */
            typedef int difference_type;

            /**
             * The type of the elements, the iterator provides access to.
             */
            typedef float64 value_type;

            /**
             * The type of a pointer to an element, the iterator provides access to.
             */
            typedef const float64* pointer;

            /**
             * The type of a reference to an element, the iterator provides access to.
             */
            typedef const float64& reference;

            /**
             * The tag that specifies the capabilities of the iterator.
             */
            typedef std::random_access_iterator_tag iterator_category;

            /**
             * Returns the element at a specific index.
             *
             * @param index The index of the element to be returned
             * @return      The element at the given index
             */
            reference operator[](uint32 index) const;

            /**
             * Returns the element, the iterator currently refers to.
             *
             * @return The element, the iterator currently refers to
             */
            reference operator*() const;

            /**
             * Returns an iterator to the next element.
             *
             * @return A reference to an iterator that refers to the next element
             */
            DenseHessianConstIterator& operator++();

            /**
             * Returns an iterator to the next element.
             *
             * @return A reference to an iterator that refers to the next element
             */
            DenseHessianConstIterator& operator++(int n);

            /**
             * Returns an iterator to the previous element.
             *
             * @return A reference to an iterator that refers to the previous element
             */
            DenseHessianConstIterator& operator--();

            /**
             * Returns an iterator to the previous element.
             *
             * @return A reference to an iterator that refers to the previous element
             */
            DenseHessianConstIterator& operator--(int n);

            /**
             * Returns whether this iterator and another one refer to the same element.
             *
             * @param rhs   A reference to another iterator
             * @return      True, if the iterators do not refer to the same element, false otherwise
             */
            bool operator!=(const DenseHessianConstIterator& rhs) const;

            /**
             * Returns whether this iterator and another one refer to the same element.
             *
             * @param rhs   A reference to another iterator
             * @return      True, if the iterators refer to the same element, false otherwise
             */
            bool operator==(const DenseHessianConstIterator& rhs) const;

            /**
             * Returns the difference between this iterator and another one.
             *
             * @param rhs   A reference to another iterator
             * @return      The difference between the iterators
             */
            difference_type operator-(const DenseHessianConstIterator& rhs) const;

    };

    /**
     * An iterator that provides access to the Hessians that are stored in a C-contiguous array and allows to modify
     * them.
     */
    class DenseHessianIterator {

        private:

            Tuple<float64>* ptr_;

        public:

            /**
             * @param ptr A pointer to an array of type `Tuple<float64>` that stores gradients and Hessians
             */
            DenseHessianIterator(Tuple<float64>* ptr);

            /**
             * The type that is used to represent the difference between two iterators.
             */
            typedef int difference_type;

            /**
             * The type of the elements, the iterator provides access to.
             */
            typedef float64 value_type;

            /**
             * The type of a pointer to an element, the iterator provides access to.
             */
            typedef float64* pointer;

            /**
             * The type of a reference to an element, the iterator provides access to.
             */
            typedef float64& reference;

            /**
             * The tag that specifies the capabilities of the iterator.
             */
            typedef std::random_access_iterator_tag iterator_category;

            /**
             * Returns the element at a specific index.
             * @param index The index of the element to be returned
             * @return      The element at the given index
             */
            reference operator[](uint32 index) const;

            /**
             * Returns the element, the iterator currently refers to.
             *
             * @return The element, the iterator currently refers to
             */
            reference operator*() const;

            /**
             * Returns an iterator to the next element.
             *
             * @return A reference to an iterator that refers to the next element
             */
            DenseHessianIterator& operator++();

            /**
             * Returns an iterator to the next element.
             *
             * @return A reference to an iterator that refers to the next element
             */
            DenseHessianIterator& operator++(int n);

            /**
             * Returns an iterator to the previous element.
             *
             * @return A reference to an iterator that refers to the previous element
             */
            DenseHessianIterator& operator--();

            /**
             * Returns an iterator to the previous element.
             *
             * @return A reference to an iterator that refers to the previous element
             */
            DenseHessianIterator& operator--(int n);

            /**
             * Returns whether this iterator and another one refer to the same element.
             *
             * @param rhs   A reference to another iterator
             * @return      True, if the iterators do not refer to the same element, false otherwise
             */
            bool operator!=(const DenseHessianIterator& rhs) const;

            /**
             * Returns whether this iterator and another one refer to the same element.
             *
             * @param rhs   A reference to another iterator
             * @return      True, if the iterators refer to the same element, false otherwise
             */
            bool operator==(const DenseHessianIterator& rhs) const;

            /**
             * Returns the difference between this iterator and another one.
             *
             * @param rhs   A reference to another iterator
             * @return      The difference between the iterators
             */
            difference_type operator-(const DenseHessianIterator& rhs) const;

    };
    
}
