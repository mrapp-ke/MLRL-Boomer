#include "mlrl/boosting/rule_model_assemblage/default_rule_auto.hpp"

namespace boosting {

    AutomaticDefaultRuleConfig::AutomaticDefaultRuleConfig(ReadableProperty<IStatisticsConfig> statisticsConfigGetter,
                                                           ReadableProperty<ILossConfig> lossConfigGetter,
                                                           ReadableProperty<IHeadConfig> headConfigGetter)
        : statisticsConfig_(statisticsConfigGetter), lossConfig_(lossConfigGetter), headConfig_(headConfigGetter) {}

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
