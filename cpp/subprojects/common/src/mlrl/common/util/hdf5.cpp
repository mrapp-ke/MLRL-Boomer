#include "hdf5.hpp"

#include <hdf5.h>

#include <stdexcept>

namespace util {

    std::string getHdf5Version() {
        unsigned int majorVersion;
        unsigned int minorVersion;
        unsigned int releaseNumber;

        if (H5get_libversion(&majorVersion, &minorVersion, &releaseNumber) < 0) {
            throw std::runtime_error("Failed to obtain the version of the HDF5 library");
        }

        return std::to_string(majorVersion) + "." + std::to_string(minorVersion) + "." + std::to_string(releaseNumber);
    }

}
