#include "hdf5_error.hpp"

#include <stdexcept>

static herr_t visitErrorStackEntry(unsigned int index, const H5E_error2_t* errorDescription, void* clientData) {
    // The error stack is traversed upwards, i.e., the first entry corresponds to the innermost error
    if (index == 0 && errorDescription->desc) {
        std::string* message = static_cast<std::string*>(clientData);

        try {
            *message = errorDescription->desc;
        } catch (...) {
            return -1;
        }
    }

    return 0;
}

void initializeHdf5() {
    // Called before each file is created or opened, rather than only once, because the setting is maintained per
    // thread by thread-safe builds of the HDF5 library
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
}

void throwHdf5Error(const std::string& operation) {
    std::string cause;
    H5Ewalk2(H5E_DEFAULT, H5E_WALK_UPWARD, &visitErrorStackEntry, &cause);
    H5Eclear2(H5E_DEFAULT);
    std::string message = "HDF5 operation failed: " + operation;

    if (!cause.empty()) {
        message += " (" + cause + ")";
    }

    throw std::runtime_error(message);
}
