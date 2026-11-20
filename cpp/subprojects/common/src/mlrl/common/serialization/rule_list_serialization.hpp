/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 *
 * Provides functions for writing rule-based models to HDF5 files and reading them from such files.
 */
#pragma once

#include "hdf5/hdf5_group.hpp"
#include "mlrl/common/model/rule_model.hpp"

#include <memory>

/**
 * The name of the group that stores a rule-based model.
 */
constexpr const char* GROUP_RULE_MODEL = "rule_model";

/**
 * Writes a rule-based model to a new group with the name `GROUP_RULE_MODEL`. Only models of type `IRuleList` are
 * supported. All rules of the model must use the same type of scores.
 *
 * @throws std::runtime_error   If the type of the model is not supported, if its rules use different types of scores,
 *                              or if the group already exists
 * @param parent                A reference to an object of type `Hdf5Group`, the new group should be added to
 * @param ruleModel             A reference to an object of type `IRuleModel` that should be written
 */
void writeRuleModel(Hdf5Group& parent, const IRuleModel& ruleModel);

/**
 * Reads a rule-based model from the group with the name `GROUP_RULE_MODEL`.
 *
 * @throws std::runtime_error   If the group does not exist or if its content is invalid
 * @param parent                A reference to an object of type `Hdf5Group` that contains the group
 * @return                      An unique pointer to an object of type `IRuleModel` that has been read. The object is of
 *                              type `IRuleList`
 */
std::unique_ptr<IRuleModel> readRuleModel(const Hdf5Group& parent);
