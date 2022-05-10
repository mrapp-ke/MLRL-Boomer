#include "boosting/rule_induction/default_rule_auto.hpp"

namespace boosting {

    AutomaticDefaultRuleConfig::AutomaticDefaultRuleConfig(
            const std::unique_ptr<IStatisticsConfig>& statisticsConfigPtr,
            const std::unique_ptr<IHeadConfig>& headConfigPtr)
        : statisticsConfigPtr_(statisticsConfigPtr), headConfigPtr_(headConfigPtr) {

    }

    bool AutomaticDefaultRuleConfig::isDefaultRuleUsed(const IRowWiseLabelMatrix& labelMatrix) const {
        if (statisticsConfigPtr_->isDense()) {
            return true;
        } else if (statisticsConfigPtr_->isSparse()) {
            return false;
        } else {
            return !shouldSparseStatisticsBePreferred(labelMatrix, false, headConfigPtr_->isPartial());
        }
    }

}
