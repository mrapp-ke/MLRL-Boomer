/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/macros.hpp"
#include "common/info.hpp"

namespace boosting {

    /**
     * Returns an object of type `ILibraryVersion` that provides information about this C++ library.
     *
     * @return An unique pointer to an object of type `ILibraryVersion`
     */
    MLRLBOOSTING_API std::unique_ptr<ILibraryInfo> getBoostingLibraryInfo();

}
