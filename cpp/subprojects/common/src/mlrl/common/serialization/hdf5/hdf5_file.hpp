/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "hdf5_group.hpp"
#include "hdf5_handle.hpp"
#include "mlrl/common/data/types.hpp"
#include "mlrl/common/util/dll_exports.hpp"

#include <cstddef>
#include <vector>

/**
 * An HDF5 file that exists only in memory and is never written to disk.
 *
 * All methods throw a `std::runtime_error` if an operation fails.
 */
class MLRLCOMMON_API Hdf5File final {
    private:

        Hdf5Handle handle_;

        explicit Hdf5File(Hdf5Handle&& handle);

    public:

        /**
         * Creates a new, empty file that can be written to.
         *
         * @return An object of type `Hdf5File` that has been created
         */
        static Hdf5File createInMemory();

        /**
         * Opens an existing file from its image, i.e., the raw bytes of the file, as returned by `getImage`. The file
         * is opened in read-only mode. The given data is copied, i.e., it does not need to outlive the file.
         *
         * @param data  A pointer to an array that stores the image of the file
         * @param size  The size of the image in bytes
         * @return      An object of type `Hdf5File` that has been opened
         */
        static Hdf5File openFromImage(const uint8* data, std::size_t size);

        /**
         * Returns the image of this file, i.e., its raw bytes.
         *
         * @return A `std::vector` that stores the image of this file
         */
        std::vector<uint8> getImage();

        /**
         * Returns the root group of this file.
         *
         * @return An object of type `Hdf5Group` that provides access to the root group
         */
        Hdf5Group getRootGroup() const;
};
