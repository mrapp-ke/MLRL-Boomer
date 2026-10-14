/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view.hpp"

#include <algorithm>
#include <utility>

/**
 * A one-dimensional view that provides access to values stored in a pre-allocated array of a specific size.
 *
 * @tparam T The type of the values, the view provides access to
 */
template<typename T>
class MLRLCOMMON_API Vector : public View<T> {
    public:

        /**
         * Allows to compute hash values for objects of type `Vector`.
         */
        struct Hash final {
            public:

                /**
                 * Computes and returns a hash value for a given object of type `Vector`.
                 *
                 * @param v A reference to an object of type `Vector`
                 * @return  The hash value
                 */
                inline std::size_t operator()(const Vector<T>& v) const {
                    typename Vector<T>::const_iterator iterator = v.cbegin();
                    uint32 numElements = v.numElements;
                    std::size_t hashValue = (std::size_t) numElements;

                    for (uint32 i = 0; i < numElements; i++) {
                        hashValue ^= iterator[i] + 0x9e3779b9 + (hashValue << 6) + (hashValue >> 2);
                    }

                    return hashValue;
                }
        };

        /**
         * Allows to check whether two objects of type `Vector` are equal or not.
         */
        struct Equal final {
            public:

                /**
                 * Returns whether two objects of type `Vector` are equal or not.
                 *
                 * @param lhs   A reference to a first object of type `Vector`
                 * @param rhs   A reference to a second object of type `Vector`
                 * @return      True, if the given objects are equal, false otherwise
                 */
                inline bool operator()(const Vector<T>& lhs, const Vector<T>& rhs) const {
                    uint32 numFirst = lhs.numElements;
                    uint32 numSecond = rhs.numElements;

                    if (numFirst != numSecond) {
                        return false;
                    }

                    typename Vector<T>::const_iterator firstIterator = lhs.cbegin();
                    typename Vector<T>::const_iterator secondIterator = rhs.cbegin();

                    for (uint32 i = 0; i < numFirst; i++) {
                        if (!isEqual(firstIterator[i], secondIterator[i])) {
                            return false;
                        }
                    }

                    return true;
                }
        };

    public:

        /**
         * The number of elements in the view.
         */
        uint32 numElements;

        /**
         * The number of elements in the view plus the number of elements added to the vector to ensure aligned memory
         * access when using SIMD operations.
         */
        uint32 numElementsWithPadding;

        /**
         * @param array         A pointer to an array of template type `T` that stores the values, the view should
         *                      provide access to
         * @param numElements   The number of elements in the view
         * @param padding       The number of elements added to the vector to ensure aligned memory access when using
         *                      SIMD operations
         */
        Vector(T* array, uint32 numElements, uint32 padding = 0)
            : View<T>(array), numElements(numElements), numElementsWithPadding(numElements + padding) {}

        /**
         * @param other A const reference to an object of type `Vector` that should be copied
         */
        Vector(const Vector<T>& other)
            : View<T>(other), numElements(other.numElements), numElementsWithPadding(other.numElementsWithPadding) {}

        /**
         * @param other A reference to an object of type `Vector` that should be moved
         */
        Vector(Vector<T>&& other)
            : View<T>(std::move(other)), numElements(other.numElements),
              numElementsWithPadding(other.numElementsWithPadding) {}

        virtual ~Vector() override {}

        /**
         * Returns a `const_iterator` to the end of the vector.
         *
         * @return A `const_iterator` to the end
         */
        typename View<T>::const_iterator cend() const {
            return &View<T>::array[numElementsWithPadding];
        }

        /**
         * Returns an `iterator` to the end of the vector.
         *
         * @return An `iterator` to the end
         */
        typename View<T>::iterator end() {
            return &View<T>::array[numElementsWithPadding];
        }

        /**
         * Sets all values stored in the view to zero.
         */
        void clear() {
            std::fill(this->begin(), this->end(), (T) 0);
        }
};

/**
 * Allocates the memory, a `Vector` provides access to.
 *
 * @tparam T                The type of the values stored in the `Vector`
 * @tparam MemoryAllocator  The type of the memory allocator to be used
 */
template<typename T, typename MemoryAllocator = DefaultMemoryAllocator>
using AllocatedVector = Allocator<Vector<T>, MemoryAllocator>;

/**
 * Allocates the memory, a `Vector` provides access to, and allows to resize it afterwards.
 *
 * @tparam View             The type of the view
 * @tparam MemoryAllocator  The type of the memory allocator to be used
 */
template<typename View, typename MemoryAllocator = DefaultMemoryAllocator>
class MLRLCOMMON_API ResizableAllocator : public Allocator<View, MemoryAllocator> {
    public:

        /**
         * The maximum number of elements in the view.
         */
        uint32 maxCapacity;

        /**
         * @param numElements   The number of elements in the view
         * @param init          True, if all elements in the view should be value-initialized, false otherwise
         */
        explicit ResizableAllocator(uint32 numElements, bool init = false)
            : Allocator<View, MemoryAllocator>(numElements, init), maxCapacity(numElements) {}

        /**
         * @param other A reference to an object of type `ResizableAllocator` that should be copied
         */
        ResizableAllocator(const ResizableAllocator<View, MemoryAllocator>& other) = delete;

        /**
         * @param other A reference to an object of type `ResizableAllocator` that should be moved
         */
        ResizableAllocator(ResizableAllocator<View, MemoryAllocator>&& other)
            : Allocator<View, MemoryAllocator>(std::move(other)), maxCapacity(other.maxCapacity) {}

        virtual ~ResizableAllocator() override {}

        /**
         * Resizes the view by re-allocating the memory it provides access to.
         *
         * @param numElements   The number of elements to which the view should be resized
         * @param freeMemory    True, if unused memory should be freed, false otherwise
         */
        void resize(uint32 numElements, bool freeMemory) {
            if (View::numElements != numElements) {
                const uint32 newPadding = MemoryAllocator::template getPadding<typename View::value_type>(numElements);

                if (numElements < maxCapacity) {
                    if (freeMemory) {
                        View::array = MemoryAllocator::reallocateMemory(
                          View::array,
                          maxCapacity + MemoryAllocator::template getPadding<typename View::value_type>(maxCapacity),
                          numElements + newPadding);
                        maxCapacity = numElements;
                    }
                } else if (numElements > maxCapacity) {
                    View::array = MemoryAllocator::reallocateMemory(
                      View::array,
                      maxCapacity + MemoryAllocator::template getPadding<typename View::value_type>(maxCapacity),
                      numElements + newPadding);
                    maxCapacity = numElements;
                }

                View::numElements = numElements;
                View::numElementsWithPadding = numElements + newPadding;
            }
        }
};

/**
 * Allocates the memory, a `Vector` provides access to, and allows to resize it afterwards.
 *
 * @tparam T                The type of the values stored in the `Vector`
 * @tparam MemoryAllocator  The type of the memory allocator to be used
 */
template<typename T, typename MemoryAllocator = DefaultMemoryAllocator>
using ResizableVector = ResizableAllocator<Vector<T>, MemoryAllocator>;

/**
 * A vector that is backed by a one-dimensional view of a specific size.
 *
 * @tparam View The type of view, the vector is backed by
 */
template<typename View>
class MLRLCOMMON_API VectorDecorator : public ViewDecorator<View> {
    public:

        /**
         * @param view The view, the vector should be backed by
         */
        explicit VectorDecorator(View&& view) : ViewDecorator<View>(std::move(view)) {}

        virtual ~VectorDecorator() override {}

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements in the vector
         */
        uint32 getNumElements() const {
            return ViewDecorator<View>::view.numElements;
        }
};

/**
 * Provides access via iterators to the values stored in a vector.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
class MLRLCOMMON_API IterableVectorDecorator : public IndexableViewDecorator<Vector> {
    public:

        /**
         * @param view The view, the vector should be backed by
         */
        explicit IterableVectorDecorator(typename Vector::view_type&& view)
            : IndexableViewDecorator<Vector>(std::move(view)) {}

        virtual ~IterableVectorDecorator() override {}

        /**
         * Returns a `const_iterator` to the end of the vector.
         *
         * @return A `const_iterator` to the end
         */
        typename IndexableViewDecorator<Vector>::const_iterator cend() const {
            return Vector::view.cend();
        }

        /**
         * Returns an `iterator` to the end of the vector.
         *
         * @return An `iterator` to the end
         */
        typename IndexableViewDecorator<Vector>::iterator end() {
            return Vector::view.end();
        }
};

/**
 * Allows to resize a vector.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
class MLRLCOMMON_API ResizableVectorDecorator : public Vector {
    public:

        /**
         * @param view The view, the vector should be backed by
         */
        explicit ResizableVectorDecorator(typename Vector::view_type&& view) : Vector(std::move(view)) {}

        virtual ~ResizableVectorDecorator() override {}

        /**
         * Sets the number of elements in the vector.
         *
         * @param numElements   The number of elements to be set
         * @param freeMemory    True, if unused memory should be freed, if possible, false otherwise
         */
        virtual void setNumElements(uint32 numElements, bool freeMemory) {
            Vector::view.resize(numElements, freeMemory);
        }
};
