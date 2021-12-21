#include "common/pruning/pruning_no.hpp"


/**
 * An implementation of the class `IPruning` that does not actually perform any pruning.
 */
class NoPruning final : public IPruning {

    public:

        std::unique_ptr<ICoverageState> prune(IThresholdsSubset& thresholdsSubset, IPartition& partition,
                                              ConditionList& conditions,
                                              const AbstractPrediction& head) const override {
            return nullptr;
        }

};

std::unique_ptr<IPruning> NoPruningFactory::create() const {
    return std::make_unique<NoPruning>();
}
