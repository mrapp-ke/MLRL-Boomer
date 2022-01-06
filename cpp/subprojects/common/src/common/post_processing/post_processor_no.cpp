#include "common/post_processing/post_processor_no.hpp"


/**
 * An implementation of the class `IPostProcessor` that does not perform any post-processing, but retains the original
 * predictions of rules.
 */
class NoPostProcessor final : virtual public IPostProcessor {

    public:

        void postProcess(AbstractPrediction& prediction) const override {
            return;
        }

};

std::unique_ptr<IPostProcessor> NoPostProcessorFactory::create() const {
    return std::make_unique<NoPostProcessor>();
}
