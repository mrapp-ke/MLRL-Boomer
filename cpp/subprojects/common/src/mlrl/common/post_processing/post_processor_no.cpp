#include "mlrl/common/post_processing/post_processor_no.hpp"

/**
 * An implementation of the class `IPostProcessor` that does not perform any post-processing, but retains the original
 * predictions of rules.
 */
class NoPostProcessor final : public IPostProcessor {
    public:

        void postProcess(View<float32>::iterator begin, View<float32>::iterator end) const override {}

        void postProcess(View<float64>::iterator begin, View<float64>::iterator end) const override {}
};

/**
 * Allows to create instances of the type `IPostProcessor` that do not perform any post-processing, but retain the
 * original predictions of rules.
 */
class NoPostProcessorFactory final : public IPostProcessorFactory {
    public:

        std::unique_ptr<IPostProcessor> create() const override {
            return std::make_unique<NoPostProcessor>();
        }
};

std::unique_ptr<IPostProcessorFactory> NoPostProcessorConfig::createPostProcessorFactory() const {
    return std::make_unique<NoPostProcessorFactory>();
}
