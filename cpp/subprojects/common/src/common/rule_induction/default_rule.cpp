#include "common/rule_induction/default_rule.hpp"


DefaultRuleConfig::DefaultRuleConfig(bool useDefaultRule)
    : useDefaultRule_(useDefaultRule) {

}

bool DefaultRuleConfig::isDefaultRuleUsed(const IRowWiseLabelMatrix& labelMatrix) const {
    return useDefaultRule_;
}
