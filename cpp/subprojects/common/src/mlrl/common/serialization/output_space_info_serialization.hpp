/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 *
 * Provides functions for writing information about the output space to HDF5 files and reading it from such files.
 */
#pragma once

#include "hdf5/hdf5_group.hpp"
#include "mlrl/common/prediction/output_space_info.hpp"

#include <memory>

/**
 * The name of the group that stores information about the output space.
 */
constexpr const char* GROUP_OUTPUT_SPACE_INFO = "output_space_info";

/**
 * Writes information about the output space to a new group with the name `GROUP_OUTPUT_SPACE_INFO`.
 *
 * @throws std::runtime_error   If the type of the information is not supported or if the group already exists
 * @param parent                A reference to an object of type `Hdf5Group`, the new group should be added to
 * @param outputSpaceInfo       A reference to an object of type `IOutputSpaceInfo` that should be written
 */
void writeOutputSpaceInfo(Hdf5Group& parent, const IOutputSpaceInfo& outputSpaceInfo);

/**
 * Reads information about the output space from the group with the name `GROUP_OUTPUT_SPACE_INFO`.
 *
 * @throws std::runtime_error   If the group does not exist or if its content is invalid
 * @param parent                A reference to an object of type `Hdf5Group` that contains the group
 * @return                      An unique pointer to an object of type `IOutputSpaceInfo` that has been read
 */
std::unique_ptr<IOutputSpaceInfo> readOutputSpaceInfo(const Hdf5Group& parent);
