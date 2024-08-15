/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view.hpp"
#include "mlrl/common/util/view_functions.hpp"

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
                    return util::hashView(v.cbegin(), v.numElements);
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
                    return util::compareViews(lhs.cbegin(), lhs.numElements, rhs.cbegin(), rhs.numElements);
                }
        };

        /**
         * The number of elements in the view.
         */
        uint32 numElements;

        /**
         * @param array         A pointer to an array of template type `T` that stores the values, the view should
         *                      provide access to
         * @param dimensions    The number of elements in each dimension of the view
         */
        Vector(T* array, std::initializer_list<uint32> dimensions)
            : View<T>(array), numElements(dimensions.begin()[0]) {}

        /**
         * @param array         A pointer to an array of template type `T` that stores the values, the view should
         *                      provide access to
         * @param numElements   The number of elements in the view
         */
        Vector(T* array, uint32 numElements) : View<T>(array), numElements(numElements) {}

        /**
         * @param other A const reference to an object of type `Vector` that should be copied
         */
        Vector(const Vector<T>& other) : View<T>(other), numElements(other.numElements) {}

        /**
         * @param other A reference to an object of type `Vector` that should be moved
         */
        Vector(Vector<T>&& other) : View<T>(std::move(other)), numElements(other.numElements) {}

        virtual ~Vector() override {}

        /**
         * Returns a `const_iterator` to the end of the vector.
         *
         * @return A `const_iterator` to the end
         */
        typename View<T>::const_iterator cend() const {
            return &View<T>::array[numElements];
        }

        /**
         * Returns an `iterator` to the end of the vector.
         *
         * @return An `iterator` to the end
         */
        typename View<T>::iterator end() {
            return &View<T>::array[numElements];
        }

        /**
         * Sets all values stored in the view to zero.
         */
        void clear() {
            util::setViewToZeros(View<T>::array, numElements);
        }
};

/**
 * Allocates the memory, a `Vector` provides access to.
 *
 * @tparam T The type of the values stored in the `Vector`
 */
template<typename T>
using AllocatedVector = Allocator<Vector<T>>;

/**
 * Allocates the memory, a `Vector` provides access to, and allows to resize it afterwards.
 *
 * @tparam T The type of the values stored in the `Vector`
 */
template<typename T>
using ResizableVector = ResizableAllocator<Vector<T>>;

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
