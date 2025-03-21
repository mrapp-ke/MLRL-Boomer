/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/library_info.hpp"
#include "mlrl/seco/util/dll_exports.hpp"

#include <memory>

namespace seco {

    /**
     * Returns an object of type `ILibraryVersion` that provides information about this C++ library.
     *
     * @return An unique pointer to an object of type `ILibraryVersion`
     */
    MLRLSECO_API std::unique_ptr<ILibraryInfo> getLibraryInfo();

}
