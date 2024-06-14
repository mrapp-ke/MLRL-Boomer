#include "mlrl/common/rule_model_assemblage/default_rule.hpp"

DefaultRuleConfig::DefaultRuleConfig(bool useDefaultRule) : useDefaultRule_(useDefaultRule) {}

bool DefaultRuleConfig::isDefaultRuleUsed(const IOutputMatrix& outputMatrix) const {
    return useDefaultRule_;
}
