/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view.hpp"

#include <unordered_set>
#include <utility>

/**
 * A view that provides access to binary values stored in a sparse vector in the dictionary of keys (DOK) format.
 */
class MLRLCOMMON_API BinaryDokVector {
    protected:

        /**
         * A pointer to an object of type `std::unordered_set` that stores the indices of all dense elements explicitly
         * stored in the view.
         */
        std::unordered_set<uint32>* indices_;

    public:

        /**
         * @param indices A pointer to an object of type `std::unordered_set` that stores the indices of all dense
         *                elements explicitly stored in the view
         */
        explicit BinaryDokVector(std::unordered_set<uint32>* indices) : indices_(indices) {}

        /**
         * @param other A reference to an object of type `BinaryDokVector` that should be copied
         */
        BinaryDokVector(const BinaryDokVector& other) : indices_(other.indices_) {}

        /**
         * @param other A reference to an object of type `BinaryDokVector` that should be moved
         */
        BinaryDokVector(BinaryDokVector&& other) : indices_(other.indices_) {}

        virtual ~BinaryDokVector() {}

        /**
         * The type of the indices, the view provides access to.
         */
        typedef uint32 index_type;

        /**
         * An iterator that provides read-only access to the indices in the view.
         */
        typedef std::unordered_set<uint32>::const_iterator index_const_iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the indices in the view.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const {
            return indices_->cbegin();
        }

        /**
         * Returns an `index_const_iterator` to the end of the indices in the view.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const {
            return indices_->cend();
        }

        /**
         * Returns the value of the element at a specific index.
         *
         * @param index The index of the element
         * @return      The value of the element at the given index
         */
        bool operator[](index_type index) const {
            return indices_->find(index) != indices_->end();
        }

        /**
         * Sets the value of the element at a specific index.
         *
         * @param index The index of the element
         * @param value The value to be set
         */
        void set(index_type index, bool value) {
            if (value) {
                indices_->insert(index);
            } else {
                indices_->erase(index);
            }
        }

        /**
         * Sets all values stored in the view to zero.
         */
        void clear() {
            indices_->clear();
        }
};

/**
 * Allocates the memory for a view that provides access to binary values stored in a sparse vector in the dictionary of
 * keys (DOK) format.
 *
 * @tparam Vector The type of the view
 */
template<typename Vector>
class MLRLCOMMON_API BinaryDokVectorAllocator : public Vector {
    public:

        BinaryDokVectorAllocator() : Vector(new std::unordered_set<typename Vector::index_type>()) {}

        /**
         * @param other A reference to an object of type `BinaryDokVectorAllocator` that should be copied
         */
        BinaryDokVectorAllocator(const BinaryDokVectorAllocator& other) : Vector(other) {
            throw std::runtime_error("Objects of type BinaryDokVectorAllocator cannot be copied");
        }

        /**
         * @param other A reference to an object of type `BinaryDokVectorAllocator` that should be moved
         */
        BinaryDokVectorAllocator(BinaryDokVectorAllocator&& other) : Vector(std::move(other)) {
            other.indices_ = nullptr;
        }

        virtual ~BinaryDokVectorAllocator() override {
            delete Vector::indices_;
        }
};

/**
 * Allocates the memory, a `BinaryDokVector` provides access to.
 */
typedef BinaryDokVectorAllocator<BinaryDokVector> AllocatedBinaryDokVector;
