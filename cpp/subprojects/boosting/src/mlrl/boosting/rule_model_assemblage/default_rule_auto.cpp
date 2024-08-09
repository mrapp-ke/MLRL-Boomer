#include "mlrl/boosting/rule_model_assemblage/default_rule_auto.hpp"

namespace boosting {

    AutomaticDefaultRuleConfig::AutomaticDefaultRuleConfig(ReadableProperty<IStatisticsConfig> statisticsConfig,
                                                           ReadableProperty<ILossConfig> lossConfig,
                                                           ReadableProperty<IHeadConfig> headConfig)
        : statisticsConfig_(statisticsConfig), lossConfig_(lossConfig), headConfig_(headConfig) {}

    bool AutomaticDefaultRuleConfig::isDefaultRuleUsed(const IOutputMatrix& outputMatrix) const {
        if (statisticsConfig_.get().isDense()) {
            return true;
        } else if (statisticsConfig_.get().isSparse()) {
            return !lossConfig_.get().isSparse();
        } else {
            return !lossConfig_.get().isSparse()
                   || !shouldSparseStatisticsBePreferred(outputMatrix, false, headConfig_.get().isPartial());
        }
    }

}
