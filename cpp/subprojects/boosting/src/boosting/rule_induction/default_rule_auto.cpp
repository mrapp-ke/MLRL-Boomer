#include "boosting/rule_induction/default_rule_auto.hpp"
#include "boosting/statistics/statistic_format.hpp"

namespace boosting {

    AutomaticDefaultRuleConfig::AutomaticDefaultRuleConfig(const std::unique_ptr<IHeadConfig>& headConfigPtr)
        : headConfigPtr_(headConfigPtr) {

    }

    bool AutomaticDefaultRuleConfig::isDefaultRuleUsed(const IRowWiseLabelMatrix& labelMatrix) const {
        return !shouldSparseStatisticsBePreferred(labelMatrix, false, headConfigPtr_->isPartial());
    }

}
