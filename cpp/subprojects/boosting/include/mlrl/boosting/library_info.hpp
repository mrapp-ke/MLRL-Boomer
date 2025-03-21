/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/util/dll_exports.hpp"
#include "mlrl/common/library_info.hpp"

#include <memory>

namespace boosting {

    /**
     * Returns an object of type `ILibraryVersion` that provides information about this C++ library.
     *
     * @return An unique pointer to an object of type `ILibraryVersion`
     */
    MLRLBOOSTING_API std::unique_ptr<ILibraryInfo> getLibraryInfo();

}
