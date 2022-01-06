/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/label_space_info.hpp"


/**
 * Defines an interface for all classes that do not provide any information about the label space.
 */
class INoLabelSpaceInfo : virtual public ILabelSpaceInfo {

    public:

        virtual ~INoLabelSpaceInfo() override { };

};

/**
 * Creates and returns a new object of the type `INoLabelSpaceInfo`.
 *
 * @return An unique pointer to an object of type `INoLabelSpaceInfo` that has been created
 */
std::unique_ptr<INoLabelSpaceInfo> createNoLabelSpaceInfo();
