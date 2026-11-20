/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 *
 * Provides functions for handling errors reported by the HDF5 library. This header includes the headers of the HDF5
 * library and must therefore only be included by source files of this library, but not by other headers.
 */
#pragma once

#include "hdf5_handle.hpp"

#include <string>
#include <type_traits>

#include <hdf5.h>

static_assert(std::is_same<hid_t, Hdf5Id>::value, "The type Hdf5Id must correspond to the type hid_t");
static_assert(std::is_same<herr_t, int>::value, "The type herr_t must correspond to the type int");

/**
 * Prepares the HDF5 library for being used by this library. In particular, it prevents the library from printing
 * error messages to `stderr`, because errors are reported via exceptions instead. Must be called before any other
 * operation of the HDF5 library is used.
 */
void initializeHdf5();

/**
 * Throws a `std::runtime_error` that describes the most recent error reported by the HDF5 library and clears the
 * library's error stack.
 *
 * @param operation A description of the operation that has failed
 */
[[noreturn]] void throwHdf5Error(const std::string& operation);

/**
 * Checks if the value returned by a function of the HDF5 library indicates an error and throws a `std::runtime_error`
 * if this is the case. By convention, all functions of the HDF5 library return negative values to indicate errors.
 *
 * @tparam T        The type of the value
 * @param result    The value returned by the function
 * @param operation A description of the operation performed by the function
 * @return          The given value
 */
template<typename T>
inline T checkHdf5(T result, const std::string& operation) {
    if (result < 0) {
        throwHdf5Error(operation);
    }

    return result;
}
