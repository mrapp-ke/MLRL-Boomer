#include "common/post_processing/post_processor_no.hpp"


void NoPostProcessor::postProcess(AbstractPrediction& prediction) const {
    return;
}

std::unique_ptr<IPostProcessor> NoPostProcessorFactory::create() const {
    return std::make_unique<NoPostProcessor>();
}
