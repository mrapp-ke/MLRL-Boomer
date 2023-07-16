/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/info.hpp"
#include "seco/macros.hpp"

namespace seco {

    /**
     * Returns an object of type `ILibraryVersion` that provides information about the C++ library "libmlrlseco".
     *
     * @return A reference to an object of type `ILibraryVersion`
     */
    MLRLSECO_API const ILibraryInfo& getSeCoLibraryInfo();

}
