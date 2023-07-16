/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/macros.hpp"
#include "common/info.hpp"

namespace boosting {

    /**
     * Returns an object of type `ILibraryVersion` that provides information about the C++ library "libmlrlboosting".
     *
     * @return A reference to an object of type `ILibraryVersion`
     */
    MLRLBOOSTING_API const ILibraryInfo& getBoostingLibraryInfo();

}
