/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/prediction/output_space_info.hpp"

#include <memory>

/**
 * Defines an interface for all classes that do not provide any information about the output space.
 */
class MLRLCOMMON_API INoOutputSpaceInfo : public IOutputSpaceInfo {
    public:

        virtual ~INoOutputSpaceInfo() override {}
};

/**
 * Creates and returns a new object of the type `INoOutputSpaceInfo`.
 *
 * @return An unique pointer to an object of type `INoOutputSpaceInfo` that has been created
 */
MLRLCOMMON_API std::unique_ptr<INoOutputSpaceInfo> createNoOutputSpaceInfo();
