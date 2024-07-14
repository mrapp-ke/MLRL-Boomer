#include "mlrl/boosting/rule_model_assemblage/default_rule_auto.hpp"

namespace boosting {

    AutomaticDefaultRuleConfig::AutomaticDefaultRuleConfig(GetterFunction<IStatisticsConfig> statisticsConfigGetter,
                                                           GetterFunction<ILossConfig> lossConfigGetter,
                                                           GetterFunction<IHeadConfig> headConfigGetter)
        : statisticsConfigGetter_(statisticsConfigGetter), lossConfigGetter_(lossConfigGetter),
          headConfigGetter_(headConfigGetter) {}

    bool AutomaticDefaultRuleConfig::isDefaultRuleUsed(const IOutputMatrix& outputMatrix) const {
        if (statisticsConfigGetter_().isDense()) {
            return true;
        } else if (statisticsConfigGetter_().isSparse()) {
            return !lossConfigGetter_().isSparse();
        } else {
            return !lossConfigGetter_().isSparse()
                   || !shouldSparseStatisticsBePreferred(outputMatrix, false, headConfigGetter_().isPartial());
        }
    }

}
