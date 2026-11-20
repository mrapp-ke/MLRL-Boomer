/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/util/dll_exports.hpp"

#include <cstdint>

/**
 * The type of identifiers that are used by the HDF5 library to refer to objects, such as files, groups, datasets or
 * attributes. It corresponds to the type `hid_t` that is defined by the HDF5 library.
 */
using Hdf5Id = std::int64_t;

/**
 * A move-only RAII wrapper that takes ownership of an identifier of an HDF5 object and closes it when it is destroyed.
 */
class MLRLCOMMON_API Hdf5Handle final {
    public:

        /**
         * A pointer to a function of the HDF5 library that closes an object with a specific identifier, e.g.,
         * `H5Fclose`, `H5Gclose` or `H5Dclose`.
         */
        typedef int (*CloseFunction)(Hdf5Id);

    private:

        Hdf5Id id_;

        CloseFunction closeFunction_;

        void close() noexcept {
            if (id_ >= 0 && closeFunction_) {
                closeFunction_(id_);
            }

            id_ = -1;
        }

    public:

        /**
         * @param id            The identifier of the HDF5 object to take ownership of
         * @param closeFunction The function that should be used for closing the object
         */
        Hdf5Handle(Hdf5Id id, CloseFunction closeFunction) noexcept : id_(id), closeFunction_(closeFunction) {}

        /**
         * @param other A reference to an object of type `Hdf5Handle` that should be copied
         */
        Hdf5Handle(const Hdf5Handle& other) = delete;

        /**
         * @param other A reference to an object of type `Hdf5Handle` that should be moved
         */
        Hdf5Handle(Hdf5Handle&& other) noexcept : id_(other.id_), closeFunction_(other.closeFunction_) {
            other.id_ = -1;
        }

        ~Hdf5Handle() {
            close();
        }

        /**
         * @param other A reference to an object of type `Hdf5Handle` that should be copied
         */
        Hdf5Handle& operator=(const Hdf5Handle& other) = delete;

        /**
         * @param other A reference to an object of type `Hdf5Handle` that should be moved
         */
        Hdf5Handle& operator=(Hdf5Handle&& other) noexcept {
            if (this != &other) {
                close();
                id_ = other.id_;
                closeFunction_ = other.closeFunction_;
                other.id_ = -1;
            }

            return *this;
        }

        /**
         * Returns the identifier of the HDF5 object.
         *
         * @return The identifier of the HDF5 object
         */
        Hdf5Id get() const noexcept {
            return id_;
        }
};
