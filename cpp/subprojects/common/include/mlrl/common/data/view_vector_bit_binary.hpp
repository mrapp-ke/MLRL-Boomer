/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector_bit.hpp"

#include <utility>

/**
 * Provides random read and write access to binary values stored in a bit vector.
 *
 * @tparam BitVector The type of view, the bit vector is backed by
 */
template<typename BitVector>
class BinaryBitVectorDecorator : public BitVector {
    public:

        /**
         * @param view The view, the bit vector should be backed by
         */
        explicit BinaryBitVectorDecorator(typename BitVector::view_type&& view) : BitVector(std::move(view)) {}

        virtual ~BinaryBitVectorDecorator() override {}

        /**
         * Returns the value of the element at a specific position.
         *
         * @param pos   The position of the element
         * @return      The value of the specified element
         */
        bool operator[](uint32 pos) const {
            return this->view.array[util::bitArrayOffset<typename BitVector::view_type::value_type>(pos)]
                   & util::bitArrayMask<typename BitVector::view_type::value_type>(pos);
        }

        /**
         * Sets a value to the element at a specific position.
         *
         * @param pos   The position of the element
         * @param value The value to be set
         */
        void set(uint32 pos, bool value) {
            if (value) {
                this->view.array[util::bitArrayOffset<typename BitVector::view_type::value_type>(pos)] |=
                  util::bitArrayMask<typename BitVector::view_type::value_type>(pos);
            } else {
                this->view.array[util::bitArrayOffset<typename BitVector::view_type::value_type>(pos)] &=
                  ~util::bitArrayMask<typename BitVector::view_type::value_type>(pos);
            }
        }
};
