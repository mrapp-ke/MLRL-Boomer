/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include <string>

namespace util {

    /**
     * Returns the version of the HDF5 library that is used at runtime.
     *
     * @return A string that specifies the version of the HDF5 library, e.g., "1.14.6"
     */
    std::string getHdf5Version();

}
