#include "common/sampling/label_sampling_no.hpp"
#include "common/indices/index_vector_full.hpp"


/**
 * An implementation of the class `ILabelSampling` that does not perform any sampling, but includes all labels.
 */
class NoLabelSubSampling final : public ILabelSampling {

    private:

        FullIndexVector indexVector_;

    public:

        /**
         * @param numLabels The total number of available labels
         */
        NoLabelSubSampling(uint32 numLabels)
            : indexVector_(numLabels) {

        }

        const IIndexVector& subSample(RNG& rng) override {
            return indexVector_;
        }

};

std::unique_ptr<ILabelSampling> NoLabelSubSamplingFactory::create(uint32 numLabels) const {
    return std::make_unique<NoLabelSubSampling>(numLabels);
}
